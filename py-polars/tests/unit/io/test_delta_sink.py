from __future__ import annotations

import json
import pickle
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from deltalake import DeltaTable, write_deltalake

import polars as pl
from polars._utils.wrap import wrap_ldf
from polars.exceptions import ComputeError, SchemaError
from polars.io.delta._sink import (
    _SINK_UUID_PROPERTY,
    DeltaSinkState,
    _format_float,
    _is_delta_type,
    _native_path,
    _partition_value,
    _relative_data_path,
    _stats_column_paths,
)
from polars.testing import assert_frame_equal

pytestmark = pytest.mark.write_disk


def _sink(lf: pl.LazyFrame, path: Path | str, mode: Any) -> None:
    DeltaSinkState.new(str(path), mode=mode).attach_sink(lf).collect()


def _read(path: Path, *sort: str) -> pl.DataFrame:
    df = pl.scan_delta(str(path)).collect()
    return df.sort(sort) if sort else df


def _adds(path: Path, version: int) -> list[dict[str, Any]]:
    log = path / "_delta_log" / f"{version:020}.json"
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    return [line["add"] for line in lines if "add" in line]


def _data_files(path: Path) -> set[str]:
    return {
        str(p.relative_to(path))
        for p in path.rglob("*.parquet")
        if "_delta_log" not in p.parts
    }


def _resolve_and_write(
    state: DeltaSinkState, lf: pl.LazyFrame
) -> list[tuple[str, int, int, bytes, pl.DataFrame | None]]:
    """Write the files of an unpartitioned sink without committing them.

    Returns the sinked files, as the native sink passes them to `commit`.
    """
    import pyarrow.parquet as pq

    wrap_ldf(state._attach_resolved_sink(lf._ldf)).collect()
    files: list[tuple[str, int, int, bytes, pl.DataFrame | None]] = []
    for p in sorted(Path(state.sink_base_path()).glob("*.parquet")):
        data = p.read_bytes()
        num_rows = pq.read_metadata(pa.BufferReader(data)).num_rows
        files.append((str(p), num_rows, len(data), data, None))
    return files


def test_delta_sink_append_round_trip(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "s": ["x", "y", None]})
    _sink(df.lazy(), tmp_path, "error")
    assert DeltaTable(str(tmp_path)).version() == 0

    _sink(df.lazy(), tmp_path, "append")

    table = DeltaTable(str(tmp_path))
    assert table.version() == 1
    history = table.history()
    assert history[0]["operation"] == "WRITE"
    assert history[1]["operation"] == "CREATE TABLE"
    assert_frame_equal(_read(tmp_path, "a", "s"), pl.concat([df, df]).sort("a", "s"))
    # Readable by delta-rs, not only Polars.
    assert table.to_pyarrow_table().num_rows == 6


def test_delta_sink_append_existing_table(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [0], "s": ["init"]}))

    df = pl.DataFrame({"a": [1, 2], "s": ["x", None]})
    _sink(df.lazy().select("s", "a"), tmp_path, "append")

    assert DeltaTable(str(tmp_path)).version() == 1
    assert _read(tmp_path, "a").to_dict(as_series=False) == {
        "a": [0, 1, 2],
        "s": ["init", "x", None],
    }


def test_delta_sink_overwrite(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1, 2, 3]}))

    _sink(pl.LazyFrame({"a": [10, 20]}), tmp_path, "overwrite")

    table = DeltaTable(str(tmp_path))
    assert table.version() == 1
    assert table.history()[0]["operationParameters"]["mode"] == "Overwrite"
    assert _read(tmp_path, "a")["a"].to_list() == [10, 20]


def test_delta_sink_overwrite_missing_table(tmp_path: Path) -> None:
    _sink(pl.LazyFrame({"a": [1]}), tmp_path, "overwrite")
    assert DeltaTable(str(tmp_path)).version() == 0
    assert _read(tmp_path)["a"].to_list() == [1]


def test_delta_sink_empty_append(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1, 2]}))

    _sink(pl.LazyFrame({"a": [1]}).clear(), tmp_path, "append")

    assert DeltaTable(str(tmp_path)).version() == 1
    # The empty file is committed, so no written file is left unreferenced.
    assert [json.loads(add["stats"])["numRecords"] for add in _adds(tmp_path, 1)] == [0]
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2]


def test_delta_sink_empty_overwrite(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1, 2]}))

    _sink(pl.LazyFrame({"a": [1]}).clear(), tmp_path, "overwrite")

    assert DeltaTable(str(tmp_path)).version() == 1
    assert [json.loads(add["stats"])["numRecords"] for add in _adds(tmp_path, 1)] == [0]
    assert _read(tmp_path).height == 0


def test_delta_sink_empty_new_table(tmp_path: Path) -> None:
    _sink(pl.LazyFrame({"a": [1]}).clear(), tmp_path, "error")

    assert [json.loads(add["stats"])["numRecords"] for add in _adds(tmp_path, 0)] == [0]
    assert _read(tmp_path).height == 0


def test_delta_sink_partitioned(tmp_path: Path) -> None:
    write_deltalake(
        str(tmp_path), pa.table({"v": [0], "k": ["init"]}), partition_by=["k"]
    )
    keys = ["a b:c#d?e+f%g/h", "ünï", None, "x", "x"]

    _sink(
        pl.LazyFrame({"v": [1, 2, 3, 4, 5], "k": keys}),
        tmp_path,
        "append",
    )

    table = DeltaTable(str(tmp_path))
    assert table.version() == 1
    assert table.metadata().partition_columns == ["k"]
    assert _read(tmp_path, "v").to_dict(as_series=False) == {
        "v": [0, 1, 2, 3, 4, 5],
        "k": ["init", *keys],
    }

    adds = _adds(tmp_path, 1)
    assert sorted(
        [a["partitionValues"]["k"] for a in adds], key=lambda v: (v is None, v)
    ) == ["a b:c#d?e+f%g/h", "x", "ünï", None]
    # One file per partition, and no partition column in the files.
    assert all(json.loads(a["stats"])["minValues"].keys() == {"v"} for a in adds)

    # delta-rs resolves the paths to the written files.
    uris = table.file_uris()
    assert len(uris) == 5
    assert table.to_pyarrow_table().num_rows == 6


def test_delta_sink_partitioned_types(tmp_path: Path) -> None:
    write_deltalake(
        str(tmp_path),
        pa.table(
            {
                "v": pa.array([0], pa.int64()),
                "d": pa.array([date(2026, 1, 1)]),
                "t": pa.array([datetime(2026, 1, 1, 12)], pa.timestamp("us")),
                "b": pa.array([True]),
                "i": pa.array([1], pa.int32()),
            }
        ),
        partition_by=["d", "t", "b", "i"],
    )

    df = pl.DataFrame(
        {
            "v": [1, 2],
            "d": [date(2026, 11, 5), None],
            "t": [datetime(2026, 11, 5, 12, 0, 0, 5), None],
            "b": [False, None],
            "i": pl.Series([-5, None], dtype=pl.Int32),
        }
    )
    _sink(df.lazy(), tmp_path, "append")

    values = sorted(
        (a["partitionValues"] for a in _adds(tmp_path, 1)),
        key=lambda pv: pv["d"] is None,
    )
    assert values == [
        {"d": "2026-11-05", "t": "2026-11-05 12:00:00.000005", "b": "false", "i": "-5"},
        {"d": None, "t": None, "b": None, "i": None},
    ]
    assert_frame_equal(
        _read(tmp_path, "v").filter(pl.col("v") > 0), df, check_column_order=False
    )


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        ("x", pl.String, "x"),
        (True, pl.Boolean, "true"),
        (-5, pl.Int8, "-5"),
        (2.0, pl.Float64, "2"),
        (-0.0, pl.Float64, "-0"),
        (1e-7, pl.Float64, "0.0000001"),
        (1e20, pl.Float64, "100000000000000000000"),
        (float("nan"), pl.Float64, "NaN"),
        (float("inf"), pl.Float64, "inf"),
        (0.1, pl.Float32, "0.1"),
        (Decimal("-0.05"), pl.Decimal(5, 2), "-0.05"),
        (Decimal("1.5"), pl.Decimal(5, 2), "1.50"),
        (date(2026, 1, 1), pl.Date, "2026-01-01"),
        (date(5, 1, 1), pl.Date, "0005-01-01"),
        (datetime(2026, 1, 1, 12), pl.Datetime("us"), "2026-01-01 12:00:00.000000"),
        (datetime(5, 1, 1, 12), pl.Datetime("us"), "0005-01-01 12:00:00.000000"),
        (
            datetime(2026, 1, 1, 13, tzinfo=timezone(timedelta(hours=1))),
            pl.Datetime("us", "UTC"),
            "2026-01-01 12:00:00.000000",
        ),
        (b"\x00\xff", pl.Binary, "\\u0000\\u00FF"),
        (None, pl.String, None),
    ],
)
def test_delta_sink_partition_value(
    value: Any, dtype: pl.DataType, expected: str | None
) -> None:
    assert _partition_value(pl.Series([value], dtype=dtype)) == expected


def test_delta_sink_partition_value_year_10000() -> None:
    # Python datetimes end at year 9999.
    value = pl.Series([253402300800000000]).cast(pl.Datetime("us"))
    assert _partition_value(value) == "+10000-01-01 00:00:00.000000"


def test_delta_sink_format_float_f32() -> None:
    assert _format_float(1e20, f32=True) == "100000000000000000000"
    assert _format_float(pa.scalar(0.1, pa.float32()).as_py(), f32=True) == "0.1"


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/t/{u}/a.parquet",
        "file:///tmp/t/{u}/a.parquet",
        "file:///C:/t/{u}/a.parquet",
        "C:\\t\\{u}\\a.parquet",
        "s3://bucket/t/{u}/a.parquet",
    ],
)
def test_delta_sink_relative_path(path: str) -> None:
    u = "0123abcd"
    assert _relative_data_path(path.format(u=u), u) == f"{u}/a.parquet"


def test_delta_sink_relative_path_outside_sink() -> None:
    with pytest.raises(ComputeError, match="not in the sink directory"):
        _relative_data_path("/tmp/t/other/a.parquet", "0123abcd")


@pytest.mark.parametrize(
    ("uri", "path"),
    [
        ("/tmp/t", "/tmp/t"),
        ("C:\\t", "C:\\t"),
        ("file:///tmp/my%20table", "/tmp/my table"),
        ("file://localhost/tmp/t", "/tmp/t"),
        ("file://server/share/t", None),
        ("s3://bucket/t", "s3://bucket/t"),
        (
            "abfss://container@account.dfs.core.windows.net/t",
            "abfss://container@account.dfs.core.windows.net/t",
        ),
        ("memory:///t", None),
        ("lakefs://repo/main/t", None),
        # No object store builder is registered for hdfs.
        ("hdfs://namenode/t", None),
        ("hf://datasets/org/t", None),
    ],
)
def test_delta_sink_native_path(uri: str, path: str | None) -> None:
    assert _native_path(uri) == path


def test_delta_sink_map_cast() -> None:
    # Schemas of data hold type instances.
    assert _is_delta_type(pl.Map(pl.String(), pl.Int64()))
    assert not _is_delta_type(pl.Map(pl.String(), pl.UInt8()))
    assert not _is_delta_type(pl.Map(pl.UInt8(), pl.String()))
    assert not _is_delta_type(pl.Map(pl.String(), pl.Datetime("ns")))


def test_delta_sink_mode_error_existing(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    files = _data_files(tmp_path)

    with pytest.raises(FileExistsError):
        _sink(pl.LazyFrame({"a": [2]}), tmp_path, "error")

    assert DeltaTable(str(tmp_path)).version() == 0
    assert _data_files(tmp_path) == files


def test_delta_sink_mode_ignore_existing(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    files = _data_files(tmp_path)

    def fail(df: pl.DataFrame) -> pl.DataFrame:
        msg = "the input of an ignored sink must not run"
        raise AssertionError(msg)

    lf = pl.LazyFrame({"a": [2]}).map_batches(fail)
    _sink(lf, tmp_path, "ignore")

    assert DeltaTable(str(tmp_path)).version() == 0
    assert _data_files(tmp_path) == files
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["_delta_log", *files])


def test_delta_sink_mode_ignore_missing(tmp_path: Path) -> None:
    _sink(pl.LazyFrame({"a": [2]}), tmp_path, "ignore")
    assert _read(tmp_path)["a"].to_list() == [2]


def test_delta_sink_commit_twice_one_version(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))

    state = DeltaSinkState.new(str(tmp_path), mode="append")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [2, 3]}))

    first = state.commit(files)
    second = state.commit(files)

    assert first.to_dict(as_series=False) == {"version": [1]}
    assert second.to_dict(as_series=False) == {"version": [1]}
    assert DeltaTable(str(tmp_path)).version() == 1
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2, 3]


def test_delta_sink_commit_version_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))

    state = DeltaSinkState.new(str(tmp_path), mode="append")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [2]}))

    # Another writer commits two versions between the sink's read and its commit.
    commit = DeltaTable.create_write_transaction

    def commit_after_others(self: DeltaTable, *args: Any, **kwargs: Any) -> None:
        write_deltalake(str(tmp_path), pa.table({"a": [10]}), mode="append")
        write_deltalake(str(tmp_path), pa.table({"a": [11]}), mode="append")
        commit(self, *args, **kwargs)

    monkeypatch.setattr(DeltaTable, "create_write_transaction", commit_after_others)

    assert state.commit(files).to_dict(as_series=False) == {"version": [3]}
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2, 10, 11]


def test_delta_sink_commit_twice_new_table(tmp_path: Path) -> None:
    state = DeltaSinkState.new(str(tmp_path), mode="error")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [2, 3]}))

    state.commit(files)
    # Mode "error" doesn't fail on its own commit.
    assert state.commit(files).to_dict(as_series=False) == {"version": [0]}
    assert DeltaTable(str(tmp_path)).version() == 0


def test_delta_sink_commit_pickled_state(tmp_path: Path) -> None:
    state = DeltaSinkState.new(str(tmp_path), mode="append")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [1]}))

    pickle.loads(pickle.dumps(state)).commit(files)

    assert _read(tmp_path)["a"].to_list() == [1]


def test_delta_sink_commit_landed_after_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    original = DeltaTable.create_write_transaction

    def commit_then_fail(self: DeltaTable, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        msg = "post-commit hook failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(DeltaTable, "create_write_transaction", commit_then_fail)

    state = DeltaSinkState.new(str(tmp_path), mode="append")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [2]}))
    assert state.commit(files).to_dict(as_series=False) == {"version": [1]}

    assert DeltaTable(str(tmp_path)).version() == 1
    assert state.remove_written_files() is not None
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2]


def test_delta_sink_commit_failed_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))

    def fail(self: DeltaTable, *args: Any, **kwargs: Any) -> None:
        msg = "commit failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(DeltaTable, "create_write_transaction", fail)

    state = DeltaSinkState.new(str(tmp_path), mode="append")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [2]}))
    with pytest.raises(RuntimeError, match="commit failed"):
        state.commit(files)

    assert Path(state.sink_base_path()).exists()
    assert state.remove_written_files() is None
    assert not Path(state.sink_base_path()).exists()
    assert DeltaTable(str(tmp_path)).version() == 0


def test_delta_sink_table_created_concurrently(tmp_path: Path) -> None:
    state = DeltaSinkState.new(str(tmp_path), mode="append")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [2]}))

    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    state.commit(files)

    assert DeltaTable(str(tmp_path)).version() == 1
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2]


def test_delta_sink_ignore_created_concurrently(tmp_path: Path) -> None:
    state = DeltaSinkState.new(str(tmp_path), mode="ignore")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [2]}))

    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    assert state.commit(files)["version"].to_list() == [None]

    assert not Path(state.sink_base_path()).exists()
    assert _read(tmp_path, "a")["a"].to_list() == [1]


def test_delta_sink_table_changed_concurrently(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    state = DeltaSinkState.new(str(tmp_path), mode="append")
    files = _resolve_and_write(state, pl.LazyFrame({"a": [2]}))

    write_deltalake(
        str(tmp_path),
        pa.table({"a": ["x"]}),
        mode="overwrite",
        schema_mode="overwrite",
    )
    with pytest.raises(SchemaError, match="schema changed"):
        state.commit(files)


def test_delta_sink_check_constraint_rejected(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    DeltaTable(str(tmp_path)).alter.add_constraint({"a_positive": "a > 0"})
    files = _data_files(tmp_path)

    with pytest.raises(NotImplementedError, match="CHECK constraints"):
        _sink(pl.LazyFrame({"a": [-1]}), tmp_path, "append")

    assert _data_files(tmp_path) == files


def test_delta_sink_append_only(tmp_path: Path) -> None:
    write_deltalake(
        str(tmp_path),
        pa.table({"a": [1]}),
        configuration={"delta.appendOnly": "true"},
    )
    files = _data_files(tmp_path)

    with pytest.raises(ValueError, match="append-only"):
        _sink(pl.LazyFrame({"a": [2]}), tmp_path, "overwrite")
    assert _data_files(tmp_path) == files

    _sink(pl.LazyFrame({"a": [2]}), tmp_path, "append")
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2]


@pytest.mark.parametrize(
    "configuration",
    [
        {"delta.enableChangeDataFeed": "true"},
    ],
)
def test_delta_sink_allowed_features(
    tmp_path: Path, configuration: dict[str, str]
) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}), configuration=configuration)
    _sink(pl.LazyFrame({"a": [2]}), tmp_path, "overwrite")
    assert _read(tmp_path)["a"].to_list() == [2]


def _add_writer_feature(path: Path, feature: str) -> None:
    """Add a writer feature to a table, also for ones deltalake can't add."""
    if feature == "domainMetadata":
        from deltalake import TableFeatures

        DeltaTable(str(path)).alter.add_feature(
            TableFeatures.DomainMetadata, allow_protocol_versions_increase=True
        )
        return
    # A reader-writer feature, written to the log by hand.
    version = DeltaTable(str(path)).version() + 1
    protocol = {
        "minReaderVersion": 3,
        "minWriterVersion": 7,
        "readerFeatures": [feature],
        "writerFeatures": [feature],
    }
    info = {"timestamp": 0, "operation": "ADD FEATURE"}
    log = path / "_delta_log" / f"{version:020}.json"
    log.write_text(
        json.dumps({"protocol": protocol}) + "\n" + json.dumps({"commitInfo": info})
    )


@pytest.mark.parametrize("feature", ["domainMetadata", "vacuumProtocolCheck"])
def test_delta_sink_feature_deltalake_rejects(tmp_path: Path, feature: str) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    _add_writer_feature(tmp_path, feature)
    files = _data_files(tmp_path)

    with pytest.raises(NotImplementedError, match=feature):
        _sink(pl.LazyFrame({"a": [2]}), tmp_path, "append")

    assert _data_files(tmp_path) == files


def test_delta_sink_variant_feature_unused(tmp_path: Path) -> None:
    from deltalake import TableFeatures

    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    DeltaTable(str(tmp_path)).alter.add_feature(
        TableFeatures.VariantType, allow_protocol_versions_increase=True
    )
    _sink(pl.LazyFrame({"a": [2]}), tmp_path, "append")
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2]


def test_delta_sink_variant_column_rejected(tmp_path: Path) -> None:
    variant = {"name": "w", "type": "variant", "nullable": True, "metadata": {}}
    schema = {
        "type": "struct",
        "fields": [
            {"name": "a", "type": "long", "nullable": True, "metadata": {}},
            {
                "name": "s",
                "type": {"type": "struct", "fields": [variant]},
                "nullable": True,
                "metadata": {},
            },
        ],
    }
    actions = [
        {
            "protocol": {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": ["variantType"],
                "writerFeatures": ["variantType"],
            }
        },
        {
            "metaData": {
                "id": "00000000-0000-0000-0000-000000000000",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps(schema),
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0,
            }
        },
    ]
    (tmp_path / "_delta_log").mkdir()
    (tmp_path / "_delta_log" / f"{0:020}.json").write_text(
        "\n".join(json.dumps(a) for a in actions)
    )

    with pytest.raises(NotImplementedError, match="variantType"):
        _sink(pl.LazyFrame({"a": [1], "s": [{"w": "x"}]}), tmp_path, "append")
    assert _data_files(tmp_path) == set()


def test_delta_sink_generated_column_rejected(tmp_path: Path) -> None:
    from deltalake.schema import Field, PrimitiveType, Schema

    DeltaTable.create(
        str(tmp_path),
        Schema(
            [
                Field("a", PrimitiveType("long")),
                Field(
                    "b",
                    PrimitiveType("long"),
                    metadata={"delta.generationExpression": "a + 1"},
                ),
            ]
        ),
    )
    with pytest.raises(NotImplementedError, match="generated"):
        _sink(pl.LazyFrame({"a": [1], "b": [2]}), tmp_path, "append")
    assert _data_files(tmp_path) == set()


def test_delta_sink_schema_mismatch(tmp_path: Path) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))

    with pytest.raises(SchemaError, match="does not match"):
        _sink(pl.LazyFrame({"a": ["x"]}), tmp_path, "append")
    with pytest.raises(SchemaError, match="does not match"):
        _sink(pl.LazyFrame({"a": [1], "b": [1]}), tmp_path, "append")
    with pytest.raises(SchemaError, match="does not match"):
        _sink(pl.LazyFrame({"a": [1]}, schema={"a": pl.Int32}), tmp_path, "append")

    # Wider data is rejected too, even when the values fit the table type.
    narrow = tmp_path / "narrow"
    write_deltalake(str(narrow), pa.table({"a": pa.array([1], pa.int32())}))
    with pytest.raises(SchemaError, match="does not match"):
        _sink(pl.LazyFrame({"a": [2]}), narrow, "append")


def test_delta_sink_non_nullable(tmp_path: Path) -> None:
    schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    write_deltalake(str(tmp_path), pa.table({"a": [1]}, schema=schema))

    _sink(pl.LazyFrame({"a": [2]}), tmp_path, "append")
    # The native sink checks the nullability of `arrow_schema`.
    with pytest.raises(SchemaError, match="nullable is false"):
        _sink(pl.LazyFrame({"a": [3, None]}), tmp_path, "append")

    assert _read(tmp_path, "a")["a"].to_list() == [1, 2]


@pytest.mark.parametrize(
    ("dtype", "valid", "invalid"),
    [
        (pa.struct([pa.field("x", pa.int64(), nullable=False)]), {"x": 1}, {"x": None}),
        (pa.list_(pa.field("element", pa.int64(), nullable=False)), [1], [1, None]),
    ],
    ids=["struct", "list"],
)
def test_delta_sink_nested_non_nullable(
    tmp_path: Path, dtype: pa.DataType, valid: Any, invalid: Any
) -> None:
    schema = pa.schema([pa.field("c", dtype)])
    write_deltalake(str(tmp_path), pa.table({"c": [valid]}, schema=schema))
    polars_dtype = pl.Schema(schema)["c"]

    with pytest.raises(SchemaError, match="nullable is false"):
        _sink(
            pl.LazyFrame({"c": [invalid]}, schema={"c": polars_dtype}),
            tmp_path,
            "append",
        )

    assert DeltaTable(str(tmp_path)).version() == 0


def _ntz_map_frame() -> pl.DataFrame:
    arr = pa.array(
        [[("a", datetime(2026, 1, 1))]], pa.map_(pa.string(), pa.timestamp("us"))
    )
    df = pl.from_arrow(pa.table({"m": arr}))
    assert isinstance(df, pl.DataFrame)
    return df


def test_delta_sink_new_table_ntz_in_map(tmp_path: Path) -> None:
    # deltalake would create the table without the timestampNtz feature.
    with pytest.raises(NotImplementedError, match="without a time zone in a map"):
        _sink(_ntz_map_frame().lazy(), tmp_path, "error")

    assert list(tmp_path.iterdir()) == []


def test_delta_sink_type_casts(tmp_path: Path) -> None:
    ts = datetime(2026, 1, 1, 12, 0, 0, 123456)
    df = pl.DataFrame(
        {
            "u8": pl.Series([1], dtype=pl.UInt8),
            "u16": pl.Series([1], dtype=pl.UInt16),
            "u32": pl.Series([1], dtype=pl.UInt32),
            "ns": pl.Series([ts], dtype=pl.Datetime("ns")),
            "ams": pl.Series([ts], dtype=pl.Datetime("ns")).dt.replace_time_zone(
                "Europe/Amsterdam"
            ),
            "cat": pl.Series(["a"], dtype=pl.Categorical),
            "enum": pl.Series(["a"], dtype=pl.Enum(["a"])),
            "nested": [{"u": 1, "l": [1]}],
        },
        schema_overrides={
            "nested": pl.Struct({"u": pl.UInt8, "l": pl.List(pl.UInt16)})
        },
    )
    _sink(df.lazy(), tmp_path, "error")

    schema = DeltaTable(str(tmp_path)).schema().to_json()
    assert '"timestamp_ntz"' in schema
    assert '"timestamp"' in schema

    expected = pl.DataFrame(
        {
            "u8": pl.Series([1], dtype=pl.Int16),
            "u16": pl.Series([1], dtype=pl.Int32),
            "u32": pl.Series([1], dtype=pl.Int64),
            "ns": pl.Series([ts], dtype=pl.Datetime("us")),
            "ams": pl.Series(
                [ts - timedelta(hours=1)], dtype=pl.Datetime("us")
            ).dt.replace_time_zone("UTC"),
            "cat": ["a"],
            "enum": ["a"],
            "nested": [{"u": 1, "l": [1]}],
        },
        schema_overrides={"nested": pl.Struct({"u": pl.Int16, "l": pl.List(pl.Int32)})},
    )
    assert_frame_equal(_read(tmp_path), expected)

    # Appending the same types to the table casts them the same way.
    _sink(df.lazy(), tmp_path, "append")
    assert _read(tmp_path).height == 2


@pytest.mark.parametrize(
    "dtype",
    [pl.UInt64, pl.Time, pl.Duration("us"), pl.Null, pl.List(pl.Duration("ms"))],
)
def test_delta_sink_unsupported_type(tmp_path: Path, dtype: pl.DataType) -> None:
    lf = pl.LazyFrame({"a": pl.Series([None], dtype=dtype)})
    with pytest.raises(TypeError):
        _sink(lf, tmp_path, "error")
    assert not (tmp_path / "_delta_log").exists()


def _stats_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "i": [1, None, 3],
            "i8": pl.Series([1, -1, None], dtype=pl.Int8),
            "f": [float("nan"), -0.0, 1.5],
            "f_nan": [float("nan"), float("nan"), None],
            "f_inf": [float("-inf"), 1.0, float("inf")],
            "f32_inf": pl.Series([1.0, float("inf"), None], dtype=pl.Float32),
            "f32": pl.Series([0.1, None, 2.5], dtype=pl.Float32),
            "s": ["x" * 100 + "z", "ü" * 70, None],
            "b": [True, None, False],
            "d": [date(2026, 1, 1), None, date(1, 1, 1)],
            "ntz": pl.Series(
                [
                    datetime(2026, 1, 1, 0, 0, 0, 123456),
                    datetime(2026, 1, 2),
                    datetime(2026, 1, 1, 0, 0, 0, 100000),
                ],
                dtype=pl.Datetime("us"),
            ),
            "tz": pl.Series(
                [datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=UTC), None, None],
                dtype=pl.Datetime("us", "UTC"),
            ),
            "dec": pl.Series(
                [Decimal("1.50"), None, Decimal("-2.25")], dtype=pl.Decimal(10, 2)
            ),
            "st": [
                {"x": 1, "y": "q", "z": {"w": 2.0}},
                None,
                {"x": 3, "y": None, "z": None},
            ],
            "l": [[1], None, [2, 3]],
            "bin": [b"\x00", None, b"zz"],
            "all_null": pl.Series([None, None, None], dtype=pl.Int64),
        }
    )


def _stats(path: Path, version: int) -> dict[str, Any]:
    (add,) = _adds(path, version)
    stats: dict[str, Any] = json.loads(add["stats"])
    return stats


def test_delta_sink_stats_parity(tmp_path: Path) -> None:
    df = _stats_data()
    reference = tmp_path / "reference"
    write_deltalake(str(reference), df.to_arrow())

    target = tmp_path / "target"
    _sink(df.lazy(), target, "error")

    expected = _stats(reference, 0)
    actual = _stats(target, 0)
    assert actual == expected
    assert "bin" not in actual["nullCount"]
    assert "l" not in actual["nullCount"]
    assert "f_nan" not in actual["minValues"]


def test_delta_sink_stats_decimal_fixed_len(tmp_path: Path) -> None:
    # Precision 38 is written as fixed-length bytes, rather than an integer.
    df = pl.DataFrame(
        {
            "d": pl.Series(
                [Decimal("1.5"), None, Decimal("-123456.78125")],
                dtype=pl.Decimal(38, 5),
            )
        }
    )
    _sink(df.lazy(), tmp_path, "error")

    stats = _stats(tmp_path, 0)
    assert stats["minValues"]["d"] == -123456.78125
    assert stats["maxValues"]["d"] == 1.5
    assert stats["nullCount"]["d"] == 1


def test_delta_sink_stats_decimal_exact(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            # The nearest float to the min is above it.
            "d38": pl.Series(
                [Decimal("1.123456789012345678"), Decimal("2")],
                dtype=pl.Decimal(38, 18),
            ),
            "d0": pl.Series(
                [Decimal("-5"), Decimal("1026600838")], dtype=pl.Decimal(10, 0)
            ),
            "small": pl.Series(
                [Decimal("0.00000001"), Decimal("-0.05")], dtype=pl.Decimal(10, 8)
            ),
        }
    )
    _sink(df.lazy(), tmp_path, "error")

    (add,) = _adds(tmp_path, 0)
    stats = json.loads(add["stats"], parse_float=str, parse_int=str)
    assert stats["minValues"] == {
        "d38": "1.123456789012345678",
        "d0": "-5",
        "small": "-0.05000000",
    }
    assert stats["maxValues"] == {
        "d38": "2.000000000000000000",
        "d0": "1026600838",
        "small": "0.00000001",
    }

    # The file is not skipped.
    out = (
        pl.scan_delta(str(tmp_path))
        .filter(pl.col("d38") == Decimal("1.123456789012345678"))
        .collect()
    )
    assert out.height == 1


def test_delta_sink_stats_row_groups(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "i": [5, None, 1, 9],
            "ntz": [
                datetime(2026, 1, 1, 0, 0, 0, 100000),
                None,
                datetime(2026, 1, 1),
                datetime(2025, 1, 1),
            ],
            "f": [1.0, float("nan"), 2.0, 3.0],
            "n": pl.Series([1, None, None, None], dtype=pl.Int64),
        }
    )
    state = DeltaSinkState.new(str(tmp_path), mode="error")
    lf = df.lazy()
    wrap_ldf(state._attach_resolved_sink(lf._ldf))  # resolve only

    import pyarrow.parquet as pq

    path = tmp_path / "rg.parquet"
    pq.write_table(df.to_arrow(), path, row_group_size=1)
    data = path.read_bytes()
    state.commit([(f"{state.sink_base_path()}rg.parquet", 4, len(data), data, None)])

    stats = _stats(tmp_path, 0)
    assert stats["minValues"]["i"] == 1
    assert stats["maxValues"]["i"] == 9
    assert stats["nullCount"]["i"] == 1
    assert stats["minValues"]["ntz"] == "2025-01-01 00:00:00"
    assert stats["maxValues"]["ntz"] == "2026-01-01 00:00:00.100"
    # A row group of only NaN has no min and max.
    assert "f" not in stats["minValues"]
    assert stats["minValues"]["n"] == stats["maxValues"]["n"] == 1
    assert stats["nullCount"]["n"] == 3


@pytest.mark.parametrize(
    "configuration",
    [
        {"delta.dataSkippingNumIndexedCols": "3"},
        {"delta.dataSkippingNumIndexedCols": "0"},
        {"delta.dataSkippingNumIndexedCols": "-1"},
        {"delta.dataSkippingStatsColumns": "s,i"},
    ],
)
def test_delta_sink_stats_columns_parity(
    tmp_path: Path, configuration: dict[str, str]
) -> None:
    df = _stats_data().select("i", "st", "s", "f", "d")

    reference = tmp_path / "reference"
    write_deltalake(str(reference), df.to_arrow(), configuration=configuration)

    target = tmp_path / "target"
    write_deltalake(str(target), df.clear().to_arrow(), configuration=configuration)
    _sink(df.lazy(), target, "append")

    assert _stats(target, 1) == _stats(reference, 0)


@pytest.mark.parametrize(
    ("stats_columns", "expected"),
    [
        ("a, s.x", [("a",), ("s", "x")]),
        ("`a.b`", [("a.b",)]),
        ("`a,b`,c", [("a,b",), ("c",)]),
        ("s.`x.y`", [("s", "x.y")]),
        ("`a``b`", [("a`b",)]),
        ("a,,b,", [("a",), ("b",)]),
    ],
)
def test_delta_sink_stats_column_paths(
    stats_columns: str, expected: list[tuple[str, ...]]
) -> None:
    assert _stats_column_paths(stats_columns) == expected


def test_delta_sink_stats_dotted_column(tmp_path: Path) -> None:
    df = pl.DataFrame({"a.b": [1, 2], "a": [{"b": 3}, {"b": 4}]})
    cases = [("`a.b`", {"a.b": 1}), ("a.b", {"a": {"b": 3}})]
    for i, (stats_columns, expected) in enumerate(cases):
        path = tmp_path / str(i)
        write_deltalake(
            str(path),
            df.clear().to_arrow(),
            configuration={"delta.dataSkippingStatsColumns": stats_columns},
        )
        _sink(df.lazy(), path, "append")
        assert _stats(path, 1)["minValues"] == expected


def test_delta_sink_deltalake_api_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(DeltaTable, "transaction_version")
    with pytest.raises(ImportError, match=r"DeltaTable\.transaction_version"):
        DeltaSinkState.new(str(tmp_path), mode="append")


def test_delta_sink_no_credential_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from polars.io.cloud.credential_provider._builder import (
        CredentialProviderBuilder,
    )

    def fail(self: CredentialProviderBuilder) -> None:
        msg = "credentials must not be resolved"
        raise AssertionError(msg)

    monkeypatch.setattr(CredentialProviderBuilder, "build_credential_provider", fail)

    state = DeltaSinkState.new(
        "s3://bucket/table",
        mode="append",
        storage_options={"aws_region": "eu-west-1"},
        credential_provider=None,
    )
    assert state.storage_options == {"aws_region": "eu-west-1"}


def _write_delta(df: pl.DataFrame, path: Path | DeltaTable, **kwargs: Any) -> None:
    df.write_delta(path if isinstance(path, DeltaTable) else str(path), **kwargs)


def _sink_delta(df: pl.DataFrame, path: Path | DeltaTable, **kwargs: Any) -> None:
    df.lazy().sink_delta(path if isinstance(path, DeltaTable) else str(path), **kwargs)


_public_writers = pytest.mark.parametrize(
    "write", [_write_delta, _sink_delta], ids=["write_delta", "sink_delta"]
)


def _native_commit(path: Path, version: int) -> bool:
    """Whether the native sink made the commit of `version`, rather than deltalake."""
    log = path / "_delta_log" / f"{version:020}.json"
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    info = next(line["commitInfo"] for line in lines if "commitInfo" in line)
    return _SINK_UUID_PROPERTY in info


@_public_writers
def test_delta_write_same_types_native(tmp_path: Path, write: Any) -> None:
    df = pl.DataFrame({"a": [1, 2], "s": ["x", None]})
    write(df, tmp_path)
    write(df.select("s", "a"), tmp_path, mode="append")
    write(df, tmp_path, mode="overwrite")

    assert [_native_commit(tmp_path, v) for v in range(3)] == [True, True, True]
    assert_frame_equal(_read(tmp_path, "a"), df)


@_public_writers
def test_delta_write_cast_deltalake(
    tmp_path: Path,
    write: Any,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": pa.array([1], pa.int32())}))
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    write(pl.DataFrame({"a": [2]}), tmp_path, mode="append")

    assert not _native_commit(tmp_path, 1)
    assert "column 'a' is Int64, and the table's is Int32" in capfd.readouterr().err
    assert_frame_equal(
        _read(tmp_path, "a"), pl.DataFrame({"a": [1, 2]}, schema={"a": pl.Int32})
    )


@_public_writers
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"a": pl.Series([1], dtype=pl.UInt32)}),
        pl.DataFrame({"a": [datetime(2026, 1, 1)]}, schema={"a": pl.Datetime("ns")}),
        pl.from_arrow(
            pa.table({"a": pa.array([[("k", 5)]], pa.map_(pa.string(), pa.uint8()))})
        ),
        pl.from_arrow(
            pa.table({"a": pa.array([[(5, "v")]], pa.map_(pa.uint8(), pa.string()))})
        ),
        pl.from_arrow(
            pa.table(
                {
                    "a": pa.array(
                        [[("k", [5])]], pa.map_(pa.string(), pa.list_(pa.uint8()))
                    )
                }
            )
        ),
    ],
    ids=["uint32", "datetime_ns", "map_value", "map_key", "map_list_value"],
)
def test_delta_write_new_table_cast_deltalake(
    tmp_path: Path, write: Any, df: pl.DataFrame
) -> None:
    write(df, tmp_path)
    assert not _native_commit(tmp_path, 0)


@_public_writers
def test_delta_write_options_deltalake(tmp_path: Path, write: Any) -> None:
    df = pl.DataFrame({"a": [1], "p": ["x"]})
    write(df, tmp_path, delta_write_options={"partition_by": ["p"]})

    assert not _native_commit(tmp_path, 0)
    assert DeltaTable(str(tmp_path)).metadata().partition_columns == ["p"]


@_public_writers
def test_delta_write_constraint_deltalake(tmp_path: Path, write: Any) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    DeltaTable(str(tmp_path)).alter.add_constraint({"a_positive": "a > 0"})

    write(pl.DataFrame({"a": [2]}), tmp_path, mode="append")

    assert not _native_commit(tmp_path, 2)
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2]


@_public_writers
def test_delta_write_error_existing_deltalake(tmp_path: Path, write: Any) -> None:
    from deltalake.exceptions import DeltaError

    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    with pytest.raises(DeltaError, match="already exists"):
        write(pl.DataFrame({"a": [2]}), tmp_path)


@_public_writers
def test_delta_write_null_removes_files(tmp_path: Path, write: Any) -> None:
    schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    write_deltalake(str(tmp_path), pa.table({"a": [1]}, schema=schema))
    entries = set(tmp_path.iterdir())

    with pytest.raises(SchemaError, match="nullable is false"):
        write(pl.DataFrame({"a": [2, None]}), tmp_path, mode="append")

    assert set(tmp_path.iterdir()) == entries


@_public_writers
def test_delta_write_null_partition_removes_files(tmp_path: Path, write: Any) -> None:
    schema = pa.schema(
        [pa.field("k", pa.string(), nullable=False), pa.field("v", pa.int64())]
    )
    write_deltalake(
        str(tmp_path), pa.table({"k": ["a"], "v": [1]}, schema=schema), partition_by="k"
    )
    entries = set(tmp_path.rglob("*"))

    # The data files don't hold partition values, so the commit checks them.
    with pytest.raises(ComputeError, match="non-nullable partition column 'k'"):
        write(pl.DataFrame({"k": ["b", None], "v": [2, 3]}), tmp_path, mode="append")

    assert set(tmp_path.rglob("*")) == entries
    assert DeltaTable(str(tmp_path)).version() == 0


@_public_writers
def test_delta_write_file_uri_percent_encoded(tmp_path: Path, write: Any) -> None:
    table = tmp_path / "my table"
    uri = table.as_uri()
    assert uri.endswith("my%20table")

    write(pl.DataFrame({"a": [1]}), uri)
    write(pl.DataFrame({"a": [2]}), uri, mode="append")

    assert _native_commit(table, 1)
    # deltalake decodes the URI, so the data files must go to `my table` too.
    assert [p.name for p in tmp_path.iterdir()] == ["my table"]
    assert sorted(DeltaTable(uri).to_pyarrow_table()["a"].to_pylist()) == [1, 2]


@_public_writers
def test_delta_write_delta_table_target(tmp_path: Path, write: Any) -> None:
    # The table URI of a DeltaTable is percent-encoded: `my%20table`.
    table = tmp_path / "my table"
    write_deltalake(str(table), pa.table({"a": [1]}))
    dt = DeltaTable(str(table))

    write(pl.DataFrame({"a": [2]}), dt, mode="append")

    assert _native_commit(table, 1)
    assert [p.name for p in tmp_path.iterdir()] == ["my table"]
    # Like `write_deltalake`, the write leaves the object at the new version.
    assert dt.version() == 1
    assert sorted(dt.to_pyarrow_table()["a"].to_pylist()) == [1, 2]


@_public_writers
def test_delta_write_new_table_ntz_in_map(tmp_path: Path, write: Any) -> None:
    from deltalake.exceptions import DeltaError

    # deltalake rejects the data before writing anything.
    with pytest.raises(DeltaError):
        write(_ntz_map_frame(), tmp_path)

    assert list(tmp_path.iterdir()) == []


@_public_writers
def test_delta_write_deltalake_api_missing(
    tmp_path: Path, write: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(DeltaTable, "transaction_version")

    write(pl.DataFrame({"a": [1]}), tmp_path)

    assert not _native_commit(tmp_path, 0)
    assert _read(tmp_path, "a")["a"].to_list() == [1]


@_public_writers
def test_delta_write_ignore(tmp_path: Path, write: Any) -> None:
    write(pl.DataFrame({"a": [1]}), tmp_path, mode="ignore")
    assert _native_commit(tmp_path, 0)
    entries = set(tmp_path.rglob("*"))

    write(pl.DataFrame({"a": [2]}), tmp_path, mode="ignore")

    assert DeltaTable(str(tmp_path)).version() == 0
    assert set(tmp_path.rglob("*")) == entries
    assert _read(tmp_path, "a")["a"].to_list() == [1]


@_public_writers
def test_delta_write_unsupported_scheme(
    tmp_path: Path, write: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Parquet sink would write the files to the local directory `memory:`.
    monkeypatch.chdir(tmp_path)
    write(pl.DataFrame({"a": [1]}), "memory:///table")
    assert list(tmp_path.iterdir()) == []


def test_delta_write_in_memory_engine(tmp_path: Path) -> None:
    pl.LazyFrame({"a": [1, 2]}).sink_delta(str(tmp_path), engine="in-memory")
    assert _native_commit(tmp_path, 0)
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2]


def _count_log_reads(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count the calls of the DeltaTable methods that read the log again."""
    counts = dict.fromkeys(["update_incremental", "transaction_version", "history"], 0)
    for name in counts:
        method = getattr(DeltaTable, name)

        def counted(
            self: DeltaTable, *a: Any, _n: str = name, _m: Any = method, **k: Any
        ) -> Any:
            counts[_n] += 1
            return _m(self, *a, **k)

        monkeypatch.setattr(DeltaTable, name, counted)
    return counts


def _before_commit(monkeypatch: pytest.MonkeyPatch, fn: Any) -> None:
    """Run `fn` after the sink wrote its files, just before it commits them."""
    commit = DeltaSinkState.commit

    def commit_after(self: DeltaSinkState, files: Any) -> Any:
        fn()
        return commit(self, files)

    monkeypatch.setattr(DeltaSinkState, "commit", commit_after)


@_public_writers
def test_delta_write_append_no_log_reads(
    tmp_path: Path, write: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    counts = _count_log_reads(monkeypatch)

    # Counted up to the commit: resolving the plan again afterwards (as the
    # schema check of the tests does) reads the log to find the commit.
    at_commit: dict[str, int] = {}
    commit = DeltaSinkState.commit

    def commit_and_count(self: DeltaSinkState, files: Any) -> Any:
        out = commit(self, files)
        at_commit.update(counts)
        return out

    monkeypatch.setattr(DeltaSinkState, "commit", commit_and_count)

    write(pl.DataFrame({"a": [2]}), tmp_path, mode="append")

    assert _native_commit(tmp_path, 1)
    assert at_commit == {
        "update_incremental": 0,
        "transaction_version": 0,
        "history": 0,
    }


@_public_writers
def test_delta_write_append_concurrent_append(
    tmp_path: Path, write: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    _before_commit(
        monkeypatch,
        lambda: write_deltalake(str(tmp_path), pa.table({"a": [10]}), mode="append"),
    )

    write(pl.DataFrame({"a": [2]}), tmp_path, mode="append")

    assert _native_commit(tmp_path, 2)
    assert _read(tmp_path, "a")["a"].to_list() == [1, 2, 10]


@_public_writers
def test_delta_write_overwrite_concurrent_append(
    tmp_path: Path, write: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    _before_commit(
        monkeypatch,
        lambda: write_deltalake(str(tmp_path), pa.table({"a": [10]}), mode="append"),
    )

    write(pl.DataFrame({"a": [2]}), tmp_path, mode="overwrite")

    assert _native_commit(tmp_path, 2)
    assert _read(tmp_path, "a")["a"].to_list() == [2]


@_public_writers
def test_delta_write_append_concurrent_schema_change(
    tmp_path: Path, write: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deltalake import Field
    from deltalake.exceptions import CommitFailedError
    from deltalake.schema import PrimitiveType

    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    _before_commit(
        monkeypatch,
        lambda: DeltaTable(str(tmp_path)).alter.add_columns(
            Field("b", PrimitiveType("string"))
        ),
    )

    entries = set(tmp_path.iterdir())

    with pytest.raises(CommitFailedError, match="Metadata changed"):
        write(pl.DataFrame({"a": [2]}), tmp_path, mode="append")

    assert DeltaTable(str(tmp_path)).version() == 1
    assert set(tmp_path.iterdir()) == entries


def _record_credential_providers(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the `credential_provider` of each `sink_parquet` call."""
    calls: list[Any] = []
    sink_parquet = pl.LazyFrame.sink_parquet

    def record(self: pl.LazyFrame, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("credential_provider", "auto"))
        return sink_parquet(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", record)
    return calls


@_public_writers
def test_delta_write_files_no_credential_provider(
    tmp_path: Path, write: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_credential_providers(monkeypatch)

    write(pl.DataFrame({"a": [1]}), tmp_path)

    assert _native_commit(tmp_path, 0)
    # The first call writes the files. The schema check of the tests resolves the
    # plan again afterwards, as a sink state that already committed.
    assert calls[0] is None


def test_delta_sink_files_auto_credential_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_credential_providers(monkeypatch)

    _sink(pl.LazyFrame({"a": [1]}), tmp_path, "error")

    assert calls[0] == "auto"


@_public_writers
@pytest.mark.parametrize("feature", ["domainMetadata", "vacuumProtocolCheck"])
def test_delta_write_feature_deltalake_rejects(
    tmp_path: Path, write: Any, feature: str
) -> None:
    from deltalake.exceptions import CommitFailedError

    write_deltalake(str(tmp_path), pa.table({"a": [1]}))
    _add_writer_feature(tmp_path, feature)
    files = _data_files(tmp_path)

    # deltalake writes the table and fails before writing any file.
    with pytest.raises(CommitFailedError, match="Unsupported table features"):
        write(pl.DataFrame({"a": [2]}), tmp_path, mode="append")

    assert _data_files(tmp_path) == files
