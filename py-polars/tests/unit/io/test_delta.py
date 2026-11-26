from __future__ import annotations

import os
import pickle
import re
import warnings
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pytest
from deltalake import DeltaTable, WriterProperties, write_deltalake
from deltalake.exceptions import DeltaError, DeltaProtocolError, TableNotFoundError
from deltalake.table import TableMerger

import polars as pl
from polars.exceptions import ArgumentRemovedError
from polars.io.cloud._utils import NoPickleOption
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,
)
from polars.io.delta._scan_resolver import (
    DeltaScanResolver,
    _source_sizes,
    _table_root,
)
from polars.io.delta._utils import (
    _extract_table_statistics_from_delta_add_actions,
    _widen_truncated_timestamp_max,
)
from polars.io.parquet.functions import scan_parquet
from polars.meta import get_index_type
from polars.testing import assert_frame_equal, assert_frame_not_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import TimeUnit
    from tests.conftest import PlMonkeyPatch


@pytest.fixture
def delta_table_path(io_files_path: Path) -> Path:
    return io_files_path / "delta-table"


def _resize_delta_add_actions(root: Path) -> None:
    """Update logged file sizes after tests rewrite Parquet data files.

    Delta files are normally immutable; these fixtures must keep sizes in sync.
    """
    import json

    for entry in sorted((root / "_delta_log").glob("*.json")):
        lines = []
        for line in entry.read_text().splitlines():
            action = json.loads(line)
            if "add" in action:
                action["add"]["size"] = (root / action["add"]["path"]).stat().st_size
                line = json.dumps(action)
            lines.append(line)
        entry.write_text("\n".join(lines) + "\n")


def new_pl_delta_dataset(source: str | DeltaTable) -> DeltaScanResolver:
    return DeltaScanResolver(
        table_=NoPickleOption(source if isinstance(source, DeltaTable) else None),
        table_uri_=source if not isinstance(source, DeltaTable) else None,
        version=None,
        storage_options=None,
        credential_provider_builder=None,
        delta_table_options=None,
        use_pyarrow=False,
        pyarrow_options=None,
    )


@pytest.mark.parametrize(
    ("uri", "table_schema"),
    [
        ("s3://bucket/literal%20space", None),
        ("s3://bucket/literal%2520space", "{}"),
        ("s3://bucket/name?query", None),
    ],
)
def test_delta_legacy_native_state_prevents_ambiguous_fallback(
    uri: str, table_schema: str | None
) -> None:
    dataset = new_pl_delta_dataset(uri)
    state = dataset.__getstate__()
    state.pop("_root_planner")
    state.pop("_table_uri_escaped")
    state.update(
        _plugin_enabled=True, _resolved_version=0, _plugin_table_schema=table_schema
    )
    dataset.__setstate__(state)

    restored = pickle.loads(pickle.dumps(dataset))
    with pytest.raises(pl.exceptions.ComputeError, match="cannot reliably reopen"):
        restored.table()


@pytest.mark.parametrize(
    ("planner", "uri_escaped", "message"),
    [
        (True, None, "cannot recover the original Delta cloud path"),
        (None, False, "cannot recover the original planner"),
    ],
)
def test_delta_rejects_unrecoverable_legacy_pinned_cloud_path(
    planner: bool | None, uri_escaped: bool | None, message: str
) -> None:
    dataset = new_pl_delta_dataset("s3://bucket/literal%20space")
    state = dataset.__getstate__()
    state.pop("_root_planner")
    state.update(
        version=0,
        _resolved_version=0,
        _plugin_enabled=planner,
        _table_uri_escaped=uri_escaped,
    )

    with pytest.raises(pl.exceptions.ComputeError, match=message):
        dataset.__setstate__(state)


@pytest.mark.parametrize("planner_saved", [False, True])
def test_delta_legacy_deltalake_state_preserves_cloud_path(
    monkeypatch: pytest.MonkeyPatch, planner_saved: bool
) -> None:
    uri = "s3://bucket/literal%20space"
    dataset = new_pl_delta_dataset(uri)
    state = dataset.__getstate__()
    state.pop("_root_planner")
    if planner_saved:
        state.pop("_table_uri_escaped")
        state["_plugin_enabled"] = False
    else:
        state["_plugin_root_requires_native"] = False
    state["_resolved_version"] = 0
    dataset.__setstate__(state)
    restored = pickle.loads(pickle.dumps(dataset))

    monkeypatch.delenv("POLARS_DELTA_PLANNER", raising=False)
    monkeypatch.setattr(
        "polars.io.delta._plugin.use_plugin_planner",
        lambda: pytest.fail("must reopen legacy deltalake URI through deltalake"),
    )
    assert restored._plugin_scan() is None
    assert restored.table_uri() == uri


@pytest.mark.parametrize(
    ("uri", "escaped", "root_planner"),
    [
        ("s3://bucket/ordinary", False, None),
        ("s3://bucket/literal%text", False, None),
        ("s3://bucket/literal%20space", True, None),
        ("s3://bucket/literal%20space", False, "deltalake"),
        ("s3://bucket/literal%2520space", True, "deltalake"),
    ],
)
def test_delta_deltalake_root_planner_tracks_native_spelling(
    monkeypatch: pytest.MonkeyPatch, uri: str, escaped: bool, root_planner: str | None
) -> None:
    dataset = new_pl_delta_dataset(uri)
    dataset._table_uri_escaped = escaped
    monkeypatch.setattr(
        "polars.io.delta._utils._get_delta_lake_table", lambda **_: object()
    )
    monkeypatch.setattr(
        "polars.io.delta._scan_resolver._check_reader_protocol", lambda _: None
    )

    dataset.table()
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._root_planner == root_planner


def test_scan_delta(delta_table_path: Path) -> None:
    ldf = pl.scan_delta(delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


def test_scan_delta_version(delta_table_path: Path) -> None:
    df1 = pl.scan_delta(delta_table_path, version=0).collect()
    df2 = pl.scan_delta(delta_table_path, version=1).collect()

    assert_frame_not_equal(df1, df2)


@pytest.mark.write_disk
def test_scan_delta_timestamp_version(tmp_path: Path) -> None:
    df_sample = pl.DataFrame({"name": ["Joey"], "age": [14]})
    df_sample.write_delta(tmp_path, mode="append")

    df_sample2 = pl.DataFrame({"name": ["Ivan"], "age": [34]})
    df_sample2.write_delta(tmp_path, mode="append")

    log_dir = tmp_path / "_delta_log"
    log_mtime_pair = [
        ("00000000000000000000.json", datetime(2010, 1, 1).timestamp()),
        ("00000000000000000001.json", datetime(2024, 1, 1).timestamp()),
    ]
    for file_name, dt_epoch in log_mtime_pair:
        file_path = log_dir / file_name
        os.utime(str(file_path), (dt_epoch, dt_epoch))

    df1 = pl.scan_delta(
        str(tmp_path), version=datetime(2010, 1, 1, tzinfo=UTC)
    ).collect()
    df2 = pl.scan_delta(
        str(tmp_path), version=datetime(2024, 1, 1, tzinfo=UTC)
    ).collect()

    assert_frame_equal(df1, df_sample)
    assert_frame_equal(df2, pl.concat([df_sample, df_sample2]), check_row_order=False)


@pytest.mark.write_disk
@pytest.mark.parametrize("planner", [None, "plugin", "deltalake"])
def test_scan_delta_timestamp_version_before_first_commit_clamps(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, planner: str | None
) -> None:
    if planner is None:
        plmonkeypatch.delenv("POLARS_DELTA_PLANNER", raising=False)
    else:
        _planner_or_skip(plmonkeypatch, planner)
    for i in range(3):
        pl.DataFrame({"a": [i]}).write_delta(tmp_path, mode="append")

    result = pl.scan_delta(tmp_path, version=datetime(1970, 1, 2, tzinfo=UTC)).collect()
    assert_frame_equal(result, pl.scan_delta(tmp_path, version=0).collect())
    assert result["a"].to_list() == [0]


def test_scan_delta_columns(delta_table_path: Path) -> None:
    ldf = pl.scan_delta(delta_table_path, version=0).select("name")

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


def test_scan_delta_polars_storage_options_keys(
    delta_table_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")
    lf = pl.scan_delta(
        delta_table_path,
        version=0,
        storage_options={
            "file_cache_ttl": 7,
            "max_retries": 3,
            "retry_timeout_ms": 9873,
            "retry_init_backoff_ms": 9874,
            "retry_max_backoff_ms": 9875,
            "retry_base_multiplier": 3.14159,
        },
        credential_provider=None,
    ).select("name")

    lf.collect()

    capture = capfd.readouterr().err

    assert "file_cache_ttl: 7" in capture

    assert (
        """\
max_retries: Some(3), \
retry_timeout: Some(9.873s), \
retry_init_backoff: Some(9.874s), \
retry_max_backoff: Some(9.875s), \
retry_base_multiplier: Some(TotalOrdWrap(3.14159)) }"""
        in capture
    )


def test_scan_delta_relative(delta_table_path: Path) -> None:
    rel_delta_table_path = str(delta_table_path / ".." / "delta-table")

    ldf = pl.scan_delta(rel_delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)

    ldf = pl.scan_delta(rel_delta_table_path, version=1)
    assert_frame_not_equal(expected, ldf.collect())


def test_read_delta(delta_table_path: Path) -> None:
    df = pl.read_delta(delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtypes=False)


def test_read_delta_version(delta_table_path: Path) -> None:
    df1 = pl.read_delta(delta_table_path, version=0)
    df2 = pl.read_delta(delta_table_path, version=1)

    assert_frame_not_equal(df1, df2)


@pytest.mark.write_disk
def test_read_delta_timestamp_version(tmp_path: Path) -> None:
    df_sample = pl.DataFrame({"name": ["Joey"], "age": [14]})
    df_sample.write_delta(tmp_path, mode="append")

    df_sample2 = pl.DataFrame({"name": ["Ivan"], "age": [34]})
    df_sample2.write_delta(tmp_path, mode="append")

    log_dir = tmp_path / "_delta_log"
    log_mtime_pair = [
        ("00000000000000000000.json", datetime(2010, 1, 1).timestamp()),
        ("00000000000000000001.json", datetime(2024, 1, 1).timestamp()),
    ]
    for file_name, dt_epoch in log_mtime_pair:
        file_path = log_dir / file_name
        os.utime(str(file_path), (dt_epoch, dt_epoch))

    df1 = pl.read_delta(str(tmp_path), version=datetime(2010, 1, 1, tzinfo=UTC))
    df2 = pl.read_delta(str(tmp_path), version=datetime(2024, 1, 1, tzinfo=UTC))

    assert_frame_equal(df1, df_sample)
    assert_frame_equal(df2, pl.concat([df_sample, df_sample2]), check_row_order=False)


def test_read_delta_columns(delta_table_path: Path) -> None:
    df = pl.read_delta(delta_table_path, version=0, columns=["name"])

    expected = pl.DataFrame({"name": ["Joey", "Ivan"]})
    assert_frame_equal(expected, df, check_dtypes=False)


def test_read_delta_relative(delta_table_path: Path) -> None:
    rel_delta_table_path = str(delta_table_path / ".." / "delta-table")

    df = pl.read_delta(rel_delta_table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtypes=False)


@pytest.mark.write_disk
def test_write_delta(df: pl.DataFrame, tmp_path: Path) -> None:
    v0 = df.select(pl.col(pl.String))
    v1 = df.select(pl.col(pl.Int64))
    df_supported = df.drop(["cat", "enum", "time"])

    # Case: Success (version 0)
    v0.write_delta(tmp_path)

    # Case: Error if table exists
    with pytest.raises(DeltaError, match="A table already exists"):
        v0.write_delta(tmp_path)

    # Case: Overwrite with new version (version 1)
    v1.write_delta(
        tmp_path, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
    )

    # Case: Error if schema contains unsupported columns
    with pytest.raises(TypeError):
        df.write_delta(
            tmp_path, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
        )

    partitioned_tbl_uri = (tmp_path / ".." / "partitioned_table").resolve()

    # Case: Write new partitioned table (version 0)
    df_supported.write_delta(
        partitioned_tbl_uri, delta_write_options={"partition_by": "strings"}
    )

    # Case: Read back
    tbl = DeltaTable(tmp_path)
    partitioned_tbl = DeltaTable(partitioned_tbl_uri)

    pl_df_0 = pl.read_delta(tbl.table_uri, version=0)
    pl_df_1 = pl.read_delta(tbl.table_uri, version=1)
    pl_df_partitioned = pl.read_delta(partitioned_tbl_uri)

    assert v0.shape == pl_df_0.shape
    assert v0.columns == pl_df_0.columns
    assert v1.shape == pl_df_1.shape
    assert v1.columns == pl_df_1.columns

    assert df_supported.shape == pl_df_partitioned.shape
    assert sorted(df_supported.columns) == sorted(pl_df_partitioned.columns)

    assert tbl.version() == 1
    assert partitioned_tbl.version() == 0

    uri = partitioned_tbl.table_uri.removeprefix("file://")
    if os.name == "nt" and uri.startswith("/"):
        uri = uri[1:]

    assert Path(uri) == partitioned_tbl_uri
    assert partitioned_tbl.metadata().partition_columns == ["strings"]

    assert_frame_equal(v0, pl_df_0, check_row_order=False)
    assert_frame_equal(v1, pl_df_1, check_row_order=False)

    cols = [c for c in df_supported.columns if not c.startswith("list_")]
    assert_frame_equal(
        df_supported.select(cols),
        pl_df_partitioned.select(cols),
        check_row_order=False,
    )

    # Case: Append to existing tables
    v1.write_delta(tmp_path, mode="append")
    tbl = DeltaTable(tmp_path)
    pl_df_1 = pl.read_delta(tbl.table_uri, version=2)

    assert tbl.version() == 2
    assert pl_df_1.shape == (6, 2)  # Rows are doubled
    assert v1.columns == pl_df_1.columns

    df_supported.write_delta(partitioned_tbl_uri, mode="append")
    partitioned_tbl = DeltaTable(partitioned_tbl_uri)
    pl_df_partitioned = pl.read_delta(partitioned_tbl.table_uri, version=1)

    assert partitioned_tbl.version() == 1
    assert pl_df_partitioned.shape == (6, 14)  # Rows are doubled
    assert sorted(df_supported.columns) == sorted(pl_df_partitioned.columns)

    df_supported.write_delta(partitioned_tbl_uri, mode="overwrite")


@pytest.mark.write_disk
def test_sink_delta(df: pl.DataFrame, tmp_path: Path) -> None:
    v0 = df.lazy().select(pl.col(pl.String))
    v1 = df.lazy().select(pl.col(pl.Int64))
    df_supported = df.drop(["cat", "enum", "time"])

    # Case: Success (version 0)
    v0.sink_delta(tmp_path)

    # Case: Error if table exists
    with pytest.raises(DeltaError, match="A table already exists"):
        v0.sink_delta(tmp_path)

    # Case: Overwrite with new version (version 1)
    v1.sink_delta(
        tmp_path, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
    )

    # Case: Error if schema contains unsupported columns
    with pytest.raises(TypeError):
        df.lazy().sink_delta(
            tmp_path, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
        )

    partitioned_tbl_uri = (tmp_path / ".." / "partitioned_table_sink").resolve()

    # Case: Write new partitioned table (version 0)
    df_supported.lazy().sink_delta(
        partitioned_tbl_uri, delta_write_options={"partition_by": "strings"}
    )

    # Case: Read back
    tbl = DeltaTable(tmp_path)
    partitioned_tbl = DeltaTable(partitioned_tbl_uri)

    pl_df_0 = pl.read_delta(tbl.table_uri, version=0)
    pl_df_1 = pl.read_delta(tbl.table_uri, version=1)
    pl_df_partitioned = pl.read_delta(partitioned_tbl_uri)

    assert v0.collect().shape == pl_df_0.shape
    assert v0.collect_schema().names() == pl_df_0.columns
    assert v1.collect().shape == pl_df_1.shape
    assert v1.collect_schema().names() == pl_df_1.columns

    assert df_supported.shape == pl_df_partitioned.shape
    assert sorted(df_supported.columns) == sorted(pl_df_partitioned.columns)

    assert tbl.version() == 1
    assert partitioned_tbl.version() == 0

    uri = partitioned_tbl.table_uri.removeprefix("file://")
    if os.name == "nt" and uri.startswith("/"):
        uri = uri[1:]

    assert Path(uri) == partitioned_tbl_uri
    assert partitioned_tbl.metadata().partition_columns == ["strings"]

    assert_frame_equal(v0.collect(), pl_df_0, check_row_order=False)
    assert_frame_equal(v1.collect(), pl_df_1, check_row_order=False)

    cols = [c for c in df_supported.columns if not c.startswith("list_")]
    assert_frame_equal(
        df_supported.select(cols),
        pl_df_partitioned.select(cols),
        check_row_order=False,
    )

    # Case: Append to existing tables
    v1.sink_delta(tmp_path, mode="append")
    tbl = DeltaTable(tmp_path)
    pl_df_1 = pl.read_delta(tbl.table_uri, version=2)

    assert tbl.version() == 2
    assert pl_df_1.shape == (6, 2)  # Rows are doubled
    assert v1.collect_schema().names() == pl_df_1.columns

    df_supported.lazy().sink_delta(partitioned_tbl_uri, mode="append")
    partitioned_tbl = DeltaTable(partitioned_tbl_uri)
    pl_df_partitioned = pl.read_delta(partitioned_tbl.table_uri, version=1)

    assert partitioned_tbl.version() == 1
    assert pl_df_partitioned.shape == (6, 14)  # Rows are doubled
    assert sorted(df_supported.columns) == sorted(pl_df_partitioned.columns)

    df_supported.lazy().sink_delta(partitioned_tbl_uri, mode="overwrite")


def test_write_delta_overwrite_schema_removed(df: pl.DataFrame) -> None:
    msg = 'Use the `delta_write_options` parameter instead and pass `{"schema_mode": "overwrite"}`.'
    with pytest.raises(ArgumentRemovedError, match=re.escape(msg)):
        df.write_delta("", mode="overwrite", overwrite_schema=True)  # type: ignore[call-overload]


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "series",
    [
        pl.Series("string", ["test"], dtype=pl.String),
        pl.Series("uint", [1], dtype=pl.UInt64),
        pl.Series("int", [1], dtype=pl.Int64),
        pl.Series(
            "uint_list",
            [[[[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]]],
            dtype=pl.List(pl.List(pl.List(pl.List(pl.UInt16)))),
        ),
        pl.Series(
            "date_ns", [datetime(2010, 1, 1, 0, 0)], dtype=pl.Datetime(time_unit="ns")
        ).dt.replace_time_zone("Australia/Lord_Howe"),
        pl.Series(
            "date_us",
            [datetime(2010, 1, 1, 0, 0)],
            dtype=pl.Datetime(time_unit="us"),
        ),
        pl.Series(
            "list_date",
            [
                [
                    datetime(2010, 1, 1, 0, 0),
                    datetime(2010, 1, 2, 0, 0),
                ]
            ],
            dtype=pl.List(pl.Datetime(time_unit="ns")),
        ),
        pl.Series(
            "list_date_us",
            [
                [
                    datetime(2010, 1, 1, 0, 0),
                    datetime(2010, 1, 2, 0, 0),
                ]
            ],
            dtype=pl.List(pl.Datetime(time_unit="ms")),
        ),
        pl.Series(
            "nested_list_date",
            [
                [
                    [
                        datetime(2010, 1, 1, 0, 0),
                        datetime(2010, 1, 2, 0, 0),
                    ]
                ]
            ],
            dtype=pl.List(pl.List(pl.Datetime(time_unit="ns"))),
        ),
        pl.Series(
            "struct_with_list",
            [
                {
                    "date_range": [
                        datetime(2010, 1, 1, 0, 0),
                        datetime(2010, 1, 2, 0, 0),
                    ],
                    "date_us": [
                        datetime(2010, 1, 1, 0, 0),
                        datetime(2010, 1, 2, 0, 0),
                    ],
                    "date_range_nested": [
                        [
                            datetime(2010, 1, 1, 0, 0),
                            datetime(2010, 1, 2, 0, 0),
                        ]
                    ],
                    "string": "test",
                    "int": 1,
                }
            ],
            dtype=pl.Struct(
                [
                    pl.Field(
                        "date_range",
                        pl.List(pl.Datetime(time_unit="ms", time_zone="UTC")),
                    ),
                    pl.Field(
                        "date_us", pl.List(pl.Datetime(time_unit="ms", time_zone=None))
                    ),
                    pl.Field(
                        "date_range_nested",
                        pl.List(pl.List(pl.Datetime(time_unit="ms", time_zone=None))),
                    ),
                    pl.Field("string", pl.String),
                    pl.Field("int", pl.UInt32),
                ]
            ),
        ),
        pl.Series(
            "list_with_struct_with_list",
            [
                [
                    {
                        "date_range": [
                            datetime(2010, 1, 1, 0, 0),
                            datetime(2010, 1, 2, 0, 0),
                        ],
                        "date_ns": [
                            datetime(2010, 1, 1, 0, 0),
                            datetime(2010, 1, 2, 0, 0),
                        ],
                        "date_range_nested": [
                            [
                                datetime(2010, 1, 1, 0, 0),
                                datetime(2010, 1, 2, 0, 0),
                            ]
                        ],
                        "string": "test",
                        "int": 1,
                    }
                ]
            ],
            dtype=pl.List(
                pl.Struct(
                    [
                        pl.Field(
                            "date_range",
                            pl.List(pl.Datetime(time_unit="ns", time_zone=None)),
                        ),
                        pl.Field(
                            "date_ns",
                            pl.List(pl.Datetime(time_unit="ns", time_zone=None)),
                        ),
                        pl.Field(
                            "date_range_nested",
                            pl.List(
                                pl.List(pl.Datetime(time_unit="ns", time_zone=None))
                            ),
                        ),
                        pl.Field("string", pl.String),
                        pl.Field("int", pl.UInt32),
                    ]
                )
            ),
        ),
    ],
)
def test_write_delta_w_compatible_schema(series: pl.Series, tmp_path: Path) -> None:
    df = series.to_frame()

    # Create table
    df.write_delta(tmp_path, mode="append")

    # Write to table again, should pass with reconstructed schema
    df.write_delta(tmp_path, mode="append")

    tbl = DeltaTable(tmp_path)
    assert tbl.version() == 1


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "expr",
    [
        pl.datetime(2010, 1, 1, time_unit="us", time_zone="UTC"),
        pl.datetime(2010, 1, 1, time_unit="ns", time_zone="America/New_York"),
        pl.datetime(2010, 1, 1, time_unit="ms", time_zone="Europe/Amsterdam"),
    ],
)
def test_write_delta_with_tz_in_df(expr: pl.Expr, tmp_path: Path) -> None:
    df = pl.select(expr)

    expected_dtype = pl.Datetime("us", "UTC")
    expected = pl.select(expr.cast(expected_dtype))

    df.write_delta(tmp_path, mode="append")
    # write second time because delta-rs also casts timestamp with tz to timestamp no tz
    df.write_delta(tmp_path, mode="append")

    # Check schema of DeltaTable object
    tbl = DeltaTable(tmp_path)
    assert pa.schema(tbl.schema().to_arrow()) == expected.to_arrow().schema

    # Check result
    result = pl.read_delta(tmp_path, version=0)
    assert_frame_equal(result, expected)


def test_write_delta_with_merge_and_no_table(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(TableNotFoundError):
        df.write_delta(
            tmp_path, mode="merge", delta_merge_options={"predicate": "a = a"}
        )


@pytest.mark.write_disk
def test_write_delta_with_merge(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    df.write_delta(tmp_path)

    merger = df.write_delta(
        tmp_path,
        mode="merge",
        delta_merge_options={
            "predicate": "s.a = t.a",
            "source_alias": "s",
            "target_alias": "t",
        },
    )

    assert isinstance(merger, TableMerger)
    assert merger._builder.source_alias == "s"
    assert merger._builder.target_alias == "t"

    merger.when_matched_delete(predicate="t.a > 2").execute()

    result = pl.read_delta(tmp_path)

    expected = df.filter(pl.col("a") <= 2)
    assert_frame_equal(result, expected, check_row_order=False)


@pytest.mark.write_disk
def test_unsupported_dtypes(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [None]}, schema={"a": pl.Null})
    with pytest.raises(TypeError, match="unsupported data type"):
        df.write_delta(tmp_path / "null")

    df = pl.DataFrame({"a": [123]}, schema={"a": pl.Time})
    with pytest.raises(TypeError, match="unsupported data type"):
        df.write_delta(tmp_path / "time")


@pytest.mark.skip(
    reason="upstream bug in delta-rs causing categorical to be written as categorical in parquet"
)
@pytest.mark.write_disk
def test_categorical_becomes_string(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": ["A", "B", "A"]}, schema={"a": pl.Categorical})
    df.write_delta(tmp_path)
    df2 = pl.read_delta(tmp_path)
    assert_frame_equal(df2, pl.DataFrame({"a": ["A", "B", "A"]}, schema={"a": pl.Utf8}))


def test_scan_delta_DT_input(delta_table_path: Path) -> None:
    DT = DeltaTable(delta_table_path, version=0)
    ldf = pl.scan_delta(DT)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, ldf.collect(), check_dtypes=False)


@pytest.mark.write_disk
def test_read_delta_empty(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = str(tmp_path)

    DeltaTable.create(path, pl.DataFrame(schema={"x": pl.Int64}).to_arrow().schema)
    assert_frame_equal(pl.read_delta(path), pl.DataFrame(schema={"x": pl.Int64}))


@pytest.mark.write_disk
def test_read_delta_arrow_map_type(tmp_path: Path) -> None:
    payload = [
        {"id": 1, "account_id": {17: "100.01.001 Cash"}},
        {"id": 2, "account_id": {18: "180.01.001 Cash", 19: "foo"}},
    ]

    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("account_id", pa.map_(pa.int32(), pa.string())),
        ]
    )
    table = pa.Table.from_pylist(payload, schema)

    expect = pl.DataFrame(table)

    table_path = str(tmp_path)
    write_deltalake(
        table_path,
        table,
        mode="overwrite",
    )

    assert_frame_equal(pl.scan_delta(table_path).collect(), expect)
    assert_frame_equal(pl.read_delta(table_path), expect)


@pytest.mark.may_fail_cloud  # reason: inspects logs
@pytest.mark.write_disk
def test_scan_delta_nanosecond_timestamp(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)]},
        schema={"timestamp": pl.Datetime("us", time_zone="UTC")},
    )

    df_nano_ts = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)]},
        schema={"timestamp": pl.Datetime("ns", time_zone=None)},
    )

    root = tmp_path / "delta"

    import deltalake

    df.write_delta(
        root,
        delta_write_options={
            "writer_properties": deltalake.WriterProperties(
                default_column_properties=deltalake.ColumnProperties(
                    statistics_enabled="NONE"
                )
            )
        },
    )

    # Manually overwrite the file with one that has nanosecond timestamps.
    parquet_files = [x for x in root.iterdir() if x.suffix == ".parquet"]
    assert len(parquet_files) == 1
    parquet_file_path = parquet_files[0]

    df_nano_ts.write_parquet(parquet_file_path)
    _resize_delta_add_actions(root)

    # Baseline: The timestamp in the file is in nanoseconds.
    q = pl.scan_parquet(parquet_file_path)
    assert q.collect_schema() == {"timestamp": pl.Datetime("ns", time_zone=None)}
    assert_frame_equal(q.collect(), df_nano_ts)

    q = pl.scan_delta(root)

    assert q.collect_schema() == {"timestamp": pl.Datetime("us", time_zone="UTC")}
    assert_frame_equal(q.collect(), df)

    # Ensure row-group skipping is functioning.
    q = pl.scan_delta(root).filter(
        pl.col("timestamp")
        < pl.lit(datetime(2025, 1, 1), dtype=pl.Datetime("us", time_zone="UTC"))
    )
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    assert_frame_equal(q.collect(), df.clear())
    assert "reading 0 / 1 row groups" in capfd.readouterr().err


@pytest.mark.write_disk
def test_scan_delta_nanosecond_timestamp_nested(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "c1": [
                {"timestamp": datetime(2025, 1, 1)},
                {"timestamp": datetime(2025, 1, 2)},
            ]
        },
        schema={"c1": pl.Struct({"timestamp": pl.Datetime("us", time_zone="UTC")})},
    )

    df_nano_ts = pl.DataFrame(
        {
            "c1": [
                {"timestamp": datetime(2025, 1, 1)},
                {"timestamp": datetime(2025, 1, 2)},
            ]
        },
        schema={"c1": pl.Struct({"timestamp": pl.Datetime("ns", time_zone=None)})},
    )

    root = tmp_path / "delta"

    df.write_delta(root)

    # Manually overwrite the file with one that has nanosecond timestamps.
    parquet_files = [x for x in root.rglob("*.parquet") if "_delta_log" not in x.parts]
    assert len(parquet_files) == 1
    parquet_file_path = parquet_files[0]

    df_nano_ts.write_parquet(parquet_file_path)
    _resize_delta_add_actions(root)

    # Baseline: The timestamp in the file is in nanoseconds.
    q = pl.scan_parquet(parquet_file_path)
    assert q.collect_schema() == {
        "c1": pl.Struct({"timestamp": pl.Datetime("ns", time_zone=None)})
    }
    assert_frame_equal(q.collect(), df_nano_ts)

    q = pl.scan_delta(root)

    assert q.collect_schema() == {
        "c1": pl.Struct({"timestamp": pl.Datetime("us", time_zone="UTC")})
    }
    assert_frame_equal(q.collect(), df)


@pytest.mark.write_disk
def test_scan_delta_schema_evolution_nested_struct_field_19915(tmp_path: Path) -> None:
    (
        pl.DataFrame(
            {"a": ["test"], "properties": [{"property_key": {"item": 1}}]}
        ).write_delta(tmp_path)
    )

    (
        pl.DataFrame(
            {
                "a": ["test1"],
                "properties": [{"property_key": {"item": 50, "item2": 10}}],
            }
        ).write_delta(
            tmp_path,
            mode="append",
            delta_write_options={"schema_mode": "merge"},
        )
    )

    q = pl.scan_delta(tmp_path)

    expect = pl.DataFrame(
        {
            "a": ["test", "test1"],
            "properties": [
                {"property_key": {"item": 1, "item2": None}},
                {"property_key": {"item": 50, "item2": 10}},
            ],
        },
        schema={
            "a": pl.String,
            "properties": pl.Struct(
                {"property_key": pl.Struct({"item": pl.Int64, "item2": pl.Int64})}
            ),
        },
    )

    assert_frame_equal(q.sort("a").collect(), expect)


@pytest.mark.write_disk
def test_scan_delta_storage_options_from_delta_table(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch
) -> None:
    import polars.io.delta._scan_resolver

    # This test inspects the deltalake planner's scan_parquet call.
    plmonkeypatch.setenv("POLARS_DELTA_PLANNER", "deltalake")
    storage_options_checked = False

    def assert_scan_parquet_storage_options(*a: Any, **kw: Any) -> Any:
        nonlocal storage_options_checked

        assert kw["storage_options"] == {
            "aws_endpoint_url": "http://localhost:777",
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
            "aws_session_token": "...",
            "endpoint_url": "...",
        }

        storage_options_checked = True

        return pl.scan_parquet(*a, **kw)

    plmonkeypatch.setattr(
        polars.io.delta._scan_resolver,
        "scan_parquet",
        assert_scan_parquet_storage_options,
    )

    df = pl.DataFrame({"a": ["test"], "properties": [{"property_key": {"item": 1}}]})

    df.write_delta(tmp_path)

    tbl = DeltaTable(
        tmp_path,
        storage_options={
            "aws_endpoint_url": "http://localhost:333",
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
            "aws_session_token": "...",
        },
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        q = pl.scan_delta(
            tbl,
            storage_options={
                "aws_endpoint_url": "http://localhost:777",
                "endpoint_url": "...",
            },
        )

        assert_frame_equal(q.collect(), df)

    assert storage_options_checked


def test_scan_delta_loads_aws_profile_endpoint_url(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    tmp_path.mkdir(exist_ok=True)

    cfg_file_path = tmp_path / "config"

    cfg_file_path.write_text("""\
[profile endpoint_333]
aws_access_key_id=A
aws_secret_access_key=A
endpoint_url = http://127.0.0.1:54321
""")

    plmonkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))
    plmonkeypatch.setenv("AWS_PROFILE", "endpoint_333")

    assert (
        builder := _init_credential_provider_builder(
            "auto", "s3://.../...", storage_options=None, caller_name="test"
        )
    ) is not None

    assert isinstance(
        provider := builder.build_credential_provider(),
        pl.CredentialProviderAWS,
    )

    assert provider._can_use_as_provider()

    assert provider._storage_update_options() == {
        "endpoint_url": "http://127.0.0.1:54321"
    }

    with pytest.raises((DeltaError, OSError), match=r"http://127.0.0.1:54321"):
        pl.scan_delta("s3://.../...").collect()

    with pytest.raises((DeltaError, OSError), match=r"http://127.0.0.1:54321"):
        pl.DataFrame({"x": 1}).write_delta("s3://.../...", mode="append")


def _df_many_types() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "p": [10, 10, 20, 20, 30, 30],
            "a": [1, 2, 3, 4, 5, None],
            "bool": [False, False, True, True, True, None],
            "int": [1, 2, 3, 4, 5, None],
            "float": [1.0, 2.0, 3.0, 4.0, 5.0, None],
            "string": ["a", "b", "c", "cc", "ccc", None],
            "struct": [
                {"x": 1, "y": 10},
                {"x": 2, "y": 20},
                {"x": 3, "y": 30},
                {"x": 4, "y": 40},
                {"x": 5, "y": 50},
                None,
            ],
        }
    ).with_columns(
        decimal=pl.col.a.cast(pl.Decimal(10, 2)),
        date=pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 6), closed="both"),
        datetime=pl.datetime_range(
            pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 6), closed="both"
        ),
    )


# TODO: uncomment dtype when fixed
@pytest.mark.parametrize(
    "expr",
    [
        # Bool (requires deltalake >= 1.5.1)
        ~pl.col.bool,
        pl.col.bool <= False,
        pl.col.bool < True,
        pl.col.bool.is_null(),
        # Integer
        pl.col.int == 2,
        pl.col.int <= 2,
        pl.col.int < 3,
        pl.col.int.is_null(),
        (pl.col.int < 2) & (pl.col.int.is_not_null()),
        # Float
        pl.col.float == 2.0,
        pl.col.float <= 2.0,
        pl.col.float < 3.0,
        pl.col.float.is_null(),
        # mixed
        (pl.col.int == 2) & (pl.col.float.is_not_null()),
        # String
        pl.col.string == "b",
        pl.col.string <= "b",
        pl.col.string.is_null(),
        # Decimal
        pl.col.decimal == pl.lit(2.0).cast(pl.Decimal(10, 2)),
        pl.col.decimal <= pl.lit(2.0).cast(pl.Decimal(10, 2)),
        pl.col.decimal < pl.lit(3.0).cast(pl.Decimal(10, 2)),
        pl.col.decimal.is_null(),
        # Struct whole-column null-count pushdown
        pl.col.struct.is_null(),
        # Struct per-field pushdown
        pl.col.struct.struct.field("x") == 2,
        pl.col.struct.struct.field("x") < 3,
        pl.col.struct.struct.field("x").is_null(),
        # whole-struct `==` is excluded: comparing structs needs ordering, which is
        # unimplemented and panics (a kernel gap, not a pushdown one). e.g.:
        # pl.col.struct == {"x": 2, "y": 20},
        # Date & datetime
        pl.col.date == pl.date(2020, 1, 1),
        pl.col.datetime == pl.datetime(2020, 1, 1),
        # on predicate
        pl.col.p == 10,
    ],
)
@pytest.mark.write_disk
def test_scan_delta_filter_delta_log_statistics_23780(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    expr: pl.Expr,
) -> None:
    df = _df_many_types()
    root = tmp_path / "delta"
    df.write_delta(root, delta_write_options={"partition_by": "p"})

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    assert_frame_equal(
        pl.scan_delta(root).filter(expr).collect(),
        df.filter(expr),
        check_column_order=False,
        check_row_order=False,
    )
    assert "skipping 2 / 3 files" in capfd.readouterr().err


@pytest.mark.write_disk
def test_scan_delta_extract_table_statistics_df(tmp_path: Path) -> None:
    import datetime as dt

    df = _df_many_types()
    root = tmp_path / "delta"
    df.write_delta(root, delta_write_options={"partition_by": "p"})

    statistics_df = _extract_table_statistics_from_delta_add_actions(
        pl.DataFrame(DeltaTable(tmp_path / "delta").get_add_actions()),
        filter_columns=df.columns,
        schema=df.schema,
        verbose=False,
    )

    assert statistics_df is not None

    assert_frame_equal(
        statistics_df,
        pl.DataFrame(
            [
                pl.Series('len', [2, 2, 2], dtype=pl.Int64),
                pl.Series('p_nc', [None, None, None], dtype=get_index_type()),
                pl.Series('p_min', [None, None, None], dtype=pl.Int64),
                pl.Series('p_max', [None, None, None], dtype=pl.Int64),
                pl.Series('a_nc', [0, 1, 0], dtype=get_index_type()),
                pl.Series('a_min', [1, 5, 3], dtype=pl.Int64),
                pl.Series('a_max', [2, 5, 4], dtype=pl.Int64),
                pl.Series('bool_nc', [0, 1, 0], dtype=get_index_type()),
                pl.Series('bool_min', [False, True, True], dtype=pl.Boolean),
                pl.Series('bool_max', [False, True, True], dtype=pl.Boolean),
                pl.Series('int_nc', [0, 1, 0], dtype=get_index_type()),
                pl.Series('int_min', [1, 5, 3], dtype=pl.Int64),
                pl.Series('int_max', [2, 5, 4], dtype=pl.Int64),
                pl.Series('float_nc', [0, 1, 0], dtype=get_index_type()),
                pl.Series('float_min', [1.0, 5.0, 3.0], dtype=pl.Float64),
                pl.Series('float_max', [2.0, 5.0, 4.0], dtype=pl.Float64),
                pl.Series('string_nc', [0, 1, 0], dtype=get_index_type()),
                pl.Series('string_min', ['a', 'ccc', 'c'], dtype=pl.String),
                pl.Series('string_max', ['b', 'ccc', 'cc'], dtype=pl.String),
                pl.Series('struct_nc', [{'x': 0, 'y': 0}, {'x': 1, 'y': 1}, {'x': 0, 'y': 0}], dtype=pl.Struct({'x': get_index_type(), 'y': get_index_type()})),
                pl.Series('struct_min', [{'x': 1, 'y': 10}, {'x': 5, 'y': 50}, {'x': 3, 'y': 30}], dtype=pl.Struct({'x': pl.Int64, 'y': pl.Int64})),
                pl.Series('struct_max', [{'x': 2, 'y': 20}, {'x': 5, 'y': 50}, {'x': 4, 'y': 40}], dtype=pl.Struct({'x': pl.Int64, 'y': pl.Int64})),
                pl.Series('decimal_nc', [0, 1, 0], dtype=get_index_type()),
                pl.Series('decimal_min', [Decimal('1.00'), Decimal('5.00'), Decimal('3.00')], dtype=pl.Decimal(precision=10, scale=2)),
                pl.Series('decimal_max', [Decimal('2.00'), Decimal('5.00'), Decimal('4.00')], dtype=pl.Decimal(precision=10, scale=2)),
                pl.Series('date_nc', [0, 0, 0], dtype=get_index_type()),
                pl.Series('date_min', [dt.date(2020, 1, 1), dt.date(2020, 1, 5), dt.date(2020, 1, 3)], dtype=pl.Date),
                pl.Series('date_max', [dt.date(2020, 1, 2), dt.date(2020, 1, 6), dt.date(2020, 1, 4)], dtype=pl.Date),
                pl.Series('datetime_nc', [0, 0, 0], dtype=get_index_type()),
                pl.Series('datetime_min', [dt.datetime(2020, 1, 1, 0, 0), dt.datetime(2020, 1, 5, 0, 0), dt.datetime(2020, 1, 3, 0, 0)], dtype=pl.Datetime(time_unit='us', time_zone=None)),
                pl.Series('datetime_max', [dt.datetime(2020, 1, 2, 0, 0, 0, 1000), dt.datetime(2020, 1, 6, 0, 0, 0, 1000), dt.datetime(2020, 1, 4, 0, 0, 0, 1000)], dtype=pl.Datetime(time_unit='us', time_zone=None)),
            ]
        ),
        check_row_order=False
    )  # fmt: skip


@pytest.mark.parametrize(
    ("expr", "n_cols", "expect_n_files_skipped"),
    [
        (pl.col.a == 2, "0", 0),
        (pl.col.a == 2, "1", 1),
        (pl.col.b == 2, "1", 0),
        (pl.col.a == 2, "2", 1),
    ],
)
@pytest.mark.write_disk
def test_scan_delta_filter_delta_log_statistics_partial_23780(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    expr: pl.Expr,
    n_cols: str,
    expect_n_files_skipped: int,
) -> None:
    df = pl.DataFrame({"p": [10, 10, 20, 20], "a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

    root = tmp_path / "delta"
    df.write_delta(
        root,
        delta_write_options={
            "partition_by": "p",
            "configuration": {
                "delta.dataSkippingNumIndexedCols": n_cols  # Disable stats collection
            },
        },
    )

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    assert_frame_equal(
        pl.scan_delta(root).filter(expr).collect(),
        df.filter(expr),
        check_column_order=False,
        check_row_order=False,
    )
    assert f"skipping {expect_n_files_skipped} / 2 files" in capfd.readouterr().err


@pytest.mark.write_disk
def test_scan_delta_filter_delta_log_statistics_delete_partition_23780(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame(
        {
            "p": [10, 10, 20, 30],
            "a": [1, 2, 3, 4],
        }
    )
    root = tmp_path / "delta"

    df.write_delta(root, delta_write_options={"partition_by": "p"})

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    expr = pl.col.a == 2
    assert_frame_equal(
        pl.scan_delta(root).filter(expr).collect(),
        df.filter(expr),
        check_column_order=False,
        check_row_order=False,
    )
    assert "skipping 2 / 3 files" in capfd.readouterr().err

    from deltalake import DeltaTable

    dt = DeltaTable(root)
    dt.delete("p = 30")

    assert_frame_equal(
        pl.scan_delta(root).filter(expr).collect(),
        df.filter(expr),
        check_column_order=False,
        check_row_order=False,
    )
    assert "skipping 1 / 2 files" in capfd.readouterr().err


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.write_disk
def test_scan_delta_use_pyarrow(tmp_path: Path, use_pyarrow: bool) -> None:
    df = pl.DataFrame({"year": [2025, 2026, 2026], "month": [0, 0, 0]})
    df.write_delta(tmp_path, delta_write_options={"partition_by": "year"})

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") == 2026)
        .collect(),
        pl.DataFrame({"year": [2026, 2026], "month": [0, 0]}),
    )

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") == 2026)
        .head(1)
        .collect(),
        pl.DataFrame({"year": [2026], "month": [0]}),
    )

    # Delta does not have stable file scan ordering.
    assert (
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow).head(1).collect().height == 1
    )


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.write_disk
def test_scan_delta_use_pyarrow_single_file(tmp_path: Path, use_pyarrow: bool) -> None:
    df = pl.DataFrame({"year": [2025, 2026, 2026], "month": [0, 0, 0]})
    df.write_delta(tmp_path)

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") == 2026)
        .collect(),
        pl.DataFrame({"year": [2026, 2026], "month": [0, 0]}),
    )

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") == 2026)
        .head(1)
        .collect(),
        pl.DataFrame({"year": [2026], "month": [0]}),
    )

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow).head(1).collect(),
        pl.DataFrame({"year": [2025], "month": [0]}),
    )

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .head(1)
        .filter(pl.col("year") == 2026)
        .collect(),
        pl.DataFrame(schema={"year": pl.Int64, "month": pl.Int64}),
    )


@pytest.mark.write_disk
def test_delta_dataset_does_not_pickle_table_object(tmp_path: Path) -> None:
    df = pl.DataFrame({"row_index": [0, 1, 2, 3, 4]})
    df.write_delta(tmp_path)

    dataset = new_pl_delta_dataset(DeltaTable(tmp_path))

    assert dataset.table_.get() is not None
    dataset = pickle.loads(pickle.dumps(dataset))
    assert dataset.table_.get() is None

    assert_frame_equal(dataset.lazy().collect(), df)


def _write_column_mapped_delta_table(root: Path, protocol: dict[str, Any]) -> None:
    """Hand-write a Delta table with column mapping mode "name"."""
    import json
    import uuid

    import pyarrow.parquet as pq

    physical_names = {"id": f"col-{uuid.uuid4()}", "v": f"col-{uuid.uuid4()}"}
    data_path = root / "part-00000.parquet"
    (root / "_delta_log").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {physical_names["id"]: [1, 2, 3], physical_names["v"]: ["a", "b", "c"]}
        ),
        data_path,
    )

    fields = [
        {
            "name": name,
            "type": dtype,
            "nullable": True,
            "metadata": {
                "delta.columnMapping.id": column_id,
                "delta.columnMapping.physicalName": physical_names[name],
            },
        }
        for column_id, (name, dtype) in enumerate([("id", "long"), ("v", "string")], 1)
    ]
    actions = [
        {"protocol": protocol},
        {
            "metaData": {
                "id": str(uuid.uuid4()),
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
                "partitionColumns": [],
                "configuration": {
                    "delta.columnMapping.mode": "name",
                    "delta.columnMapping.maxColumnId": "2",
                },
                "createdTime": 0,
            }
        },
        {
            "add": {
                "path": data_path.name,
                "partitionValues": {},
                "size": data_path.stat().st_size,
                "modificationTime": 0,
                "dataChange": True,
            }
        },
    ]
    (root / "_delta_log" / "00000000000000000000.json").write_text(
        "".join(json.dumps(action) + "\n" for action in actions)
    )


@pytest.mark.parametrize(
    ("protocol", "match"),
    [
        (
            {"minReaderVersion": 2, "minWriterVersion": 5},
            r"reader version 2 \(column mapping\)",
        ),
        (
            {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": ["columnMapping"],
                "writerFeatures": ["columnMapping"],
            },
            "columnMapping",
        ),
    ],
    ids=["reader_v2", "reader_v3_feature"],
)
@pytest.mark.parametrize("use_pyarrow", [False, True])
@pytest.mark.parametrize("as_delta_table", [False, True])
@pytest.mark.write_disk
def test_scan_delta_column_mapping_raises_30179(
    tmp_path: Path,
    protocol: dict[str, Any],
    match: str,
    use_pyarrow: bool,
    as_delta_table: bool,
) -> None:
    _write_column_mapped_delta_table(tmp_path, protocol)
    source = DeltaTable(tmp_path) if as_delta_table else str(tmp_path)

    from polars.io.delta._plugin import use_plugin_planner

    plugin = use_plugin_planner()
    expected = pl.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]})
    if plugin and not use_pyarrow:
        assert_frame_equal(pl.read_delta(source), expected)
        assert_frame_equal(
            pl.scan_delta(source).filter(pl.col("id") == 2).collect(),
            expected.filter(pl.col("id") == 2),
        )
    else:
        with pytest.raises(DeltaProtocolError, match=match):
            pl.read_delta(source, use_pyarrow=use_pyarrow)
        with pytest.raises(DeltaProtocolError, match=match):
            pl.scan_delta(source, use_pyarrow=use_pyarrow).filter(
                pl.col("id") == 2
            ).collect()

    # Pickling must not run the check; the reloaded table uses the selected planner.
    dataset = pickle.loads(pickle.dumps(new_pl_delta_dataset(source)))
    if plugin:
        assert dataset.schema() == expected.schema
    else:
        with pytest.raises(DeltaProtocolError, match=match):
            dataset.schema()


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.write_disk
def test_delta_partition_filter(tmp_path: Path, use_pyarrow: bool) -> None:
    df = pl.DataFrame({"row_index": [0, 1, 2, 3, 4], "year": 2026})
    df.write_delta(tmp_path, delta_write_options={"partition_by": "year"})

    for path in DeltaTable(tmp_path).file_uris():
        Path(path).unlink()

    with pytest.raises((FileNotFoundError, OSError)):
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow).collect()

    assert_frame_equal(
        pl.scan_delta(tmp_path, use_pyarrow=use_pyarrow)
        .filter(pl.col("year") < 0)
        .collect(),
        pl.DataFrame(schema=df.schema),
    )


@pytest.mark.write_disk
@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_scan_delta_collect_without_version_scans_latest(
    tmp_path: Path,
    use_pyarrow: bool,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    pl.DataFrame({"a": [0]}).write_delta(tmp_path)
    table = DeltaTable(tmp_path)

    q = pl.scan_delta(table, use_pyarrow=use_pyarrow)

    assert_frame_equal(q.collect(), pl.DataFrame({"a": [0]}))

    pl.DataFrame({"a": [1]}).write_delta(table, mode="append")

    assert_frame_equal(q.collect().sort("*"), pl.DataFrame({"a": [0, 1]}))

    version = table.version()

    q_with_id = pl.scan_delta(table, use_pyarrow=use_pyarrow, version=version)

    assert_frame_equal(q_with_id.collect().sort("*"), pl.DataFrame({"a": [0, 1]}))

    pl.DataFrame({"a": [2]}).write_delta(table, mode="append")

    assert_frame_equal(q.collect().sort("*"), pl.DataFrame({"a": [0, 1, 2]}))

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    assert_frame_equal(q_with_id.collect().sort("*"), pl.DataFrame({"a": [0, 1]}))

    capture = capfd.readouterr().err

    assert (
        "DeltaScanResolver: resolve_lazyframe(): early return (version_key = '1')"
        in capture
    )


@pytest.mark.write_disk
def test_scan_delta_filter_delta_log_statistics_missing_26444(tmp_path: Path) -> None:
    pl.DataFrame({"x": [1, 2], "y": [True, False]}).write_delta(tmp_path)

    assert_frame_equal(
        pl.scan_delta(tmp_path).filter("y").collect(),
        pl.DataFrame({"x": 1, "y": True}),
    )

    schema = {
        "bool": pl.Boolean,
        "string": pl.String,
        "binary": pl.Binary,
        "int8": pl.Int8,
        "null": pl.Null,
    }

    for actions_df in [
        pl.DataFrame({"num_records": [1, 2, 3]}),
        pl.DataFrame({"num_records": [1, 2, 3], "min": [{}, {}, {}]}),
        pl.DataFrame({"num_records": [1, 2, 3], "max": [{}, {}, {}]}),
        pl.DataFrame({"num_records": [1, 2, 3], "null_count": [{}, {}, {}]}),
    ]:
        df = _extract_table_statistics_from_delta_add_actions(
            actions_df,
            filter_columns=[*schema],
            schema=schema,
            verbose=False,
        )

        assert df is not None

        assert_frame_equal(
            df,
            pl.DataFrame(
                [
                    pl.Series("len", [1, 2, 3], dtype=pl.Int64),
                    pl.Series("bool_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("bool_min", [None, None, None], dtype=pl.Boolean),
                    pl.Series("bool_max", [None, None, None], dtype=pl.Boolean),
                    pl.Series("string_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("string_min", [None, None, None], dtype=pl.String),
                    pl.Series("string_max", [None, None, None], dtype=pl.String),
                    pl.Series("binary_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("binary_min", [None, None, None], dtype=pl.Binary),
                    pl.Series("binary_max", [None, None, None], dtype=pl.Binary),
                    pl.Series("int8_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("int8_min", [None, None, None], dtype=pl.Int8),
                    pl.Series("int8_max", [None, None, None], dtype=pl.Int8),
                    pl.Series("null_nc", [None, None, None], dtype=pl.UInt32),
                    pl.Series("null_min", [None, None, None], dtype=pl.Null),
                    pl.Series("null_max", [None, None, None], dtype=pl.Null),
                ]
            ),
        )

        assert (
            _extract_table_statistics_from_delta_add_actions(
                pl.DataFrame(),
                filter_columns=[*schema],
                schema=schema,
                verbose=False,
            )
            is None
        )


@pytest.mark.write_disk
def test_scan_delta_filter_combined_predicates_statistics_27072(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    df = pl.DataFrame({"p": [10, 10, 20, 20, 30, 30]})

    dfs = [df.with_columns(pl.lit(i).alias("a")) for i in range(3)]

    root = tmp_path / "delta"
    for df in dfs:
        df.write_delta(root, delta_write_options={"partition_by": "p"}, mode="append")

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()

    filter = (pl.col("p") == 10) & (pl.col("a") == 1)

    assert_frame_equal(
        pl.scan_delta(root).filter(filter).collect(),
        pl.concat(dfs).filter(filter),
        check_column_order=False,
        check_row_order=False,
    )
    assert "skipping 8 / 9 files" in capfd.readouterr().err


@pytest.mark.write_disk
def test_scan_delta_literal_filter_empty_df_27242(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    df.write_delta(tmp_path)

    out = pl.scan_delta(tmp_path).filter(pl.lit(True)).collect()
    assert_frame_equal(df, out)


@pytest.mark.write_disk
def test_scan_delta_use_pyarrow_float_literals(tmp_path: Path) -> None:
    df = pl.DataFrame({"f": [1.0, 2.5, 1e20]})
    df.write_delta(tmp_path)

    # Floats are rendered as Python float literals, also when integral (`1.0`, `1e20`).
    for predicate in [
        pl.col("f").is_in([1.0, 1e20]),
        pl.col("f") == 1e20,
        pl.col("f") == 1.0,
    ]:
        assert_frame_equal(
            pl.scan_delta(tmp_path, use_pyarrow=True).filter(predicate).collect(),
            df.filter(predicate),
            check_row_order=False,
        )


def test_scan_delta_predicate_pushdown_struct_column_27857(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "error_flags": pl.Series(
                [
                    {"length_issue": True, "incomplete": True},
                    {"length_issue": False, "incomplete": False},
                ]
            )
        }
    )
    df.write_delta(tmp_path)

    predicate = pl.any_horizontal(pl.col("error_flags").struct.unnest())
    out = pl.scan_delta(tmp_path).filter(predicate).collect()

    expected = pl.DataFrame(
        {"error_flags": pl.Series([{"length_issue": True, "incomplete": True}])}
    )
    assert_frame_equal(out, expected)


@pytest.mark.write_disk
def test_scan_delta_filter_truncated_timestamp_max_statistics_30147(
    tmp_path: Path,
) -> None:
    # The Delta protocol truncates timestamp min/max statistics to milliseconds, so
    # the max in the log can be below the real max of the file.
    first = datetime(2026, 1, 1, 0, 0, 0, 999)
    second = datetime(2026, 1, 1, 1, 0, 0)

    pl.DataFrame({"ts": [first], "s": [{"ts": first}]}).write_delta(tmp_path)
    pl.DataFrame({"ts": [second], "s": [{"ts": second}]}).write_delta(
        tmp_path, mode="append"
    )

    q = pl.scan_delta(tmp_path)
    ts = pl.col("ts")
    s_ts = pl.col("s").struct.field("ts")

    assert q.filter(ts == first).collect().height == 1
    assert q.filter(ts >= first).collect().height == 2
    assert q.filter(ts > first).collect().height == 1
    assert q.filter(ts.is_in([first])).collect().height == 1
    assert q.filter(s_ts == first).collect().height == 1
    assert q.filter(s_ts >= first).collect().height == 2


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_widen_truncated_timestamp_max(time_unit: TimeUnit) -> None:
    one_ms = {"ms": 1, "us": 1_000, "ns": 1_000_000}[time_unit]
    dtype = pl.Datetime(time_unit, "UTC")
    i64_max = 2**63 - 1

    s = pl.Series("a", [0, i64_max - one_ms, i64_max, None]).cast(dtype)
    out = _widen_truncated_timestamp_max(s)

    # Keeps the time unit and maps overflow to null (unknown) instead of wrapping.
    assert out.dtype == dtype
    assert out.to_physical().to_list() == [one_ms, i64_max, None, None]

    struct_dtype = pl.Struct({"x": pl.Struct({"ts": dtype}), "y": pl.Int64})
    s = pl.Series(
        "s",
        [{"x": {"ts": 0}, "y": 0}, None, {"x": None, "y": None}],
        dtype=pl.Struct({"x": pl.Struct({"ts": pl.Int64}), "y": pl.Int64}),
    ).cast(struct_dtype)
    out = _widen_truncated_timestamp_max(s)

    assert out.dtype == struct_dtype
    assert out.struct.field("x").struct.field("ts").to_physical().to_list() == [
        one_ms,
        None,
        None,
    ]
    assert out.is_null().to_list() == [False, True, False]


@pytest.mark.write_disk
def test_scan_delta_predicate_pushdown_struct_is_not_null(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # A struct row with a null *field* is still a non-null *struct*, so `is_not_null()`
    # must keep it. The per-field null count must not be read as the struct's row-level
    # null count, or it would wrongly skip the first file (`a` all-null, structs valid).
    st = pl.Struct({"a": pl.Int64, "b": pl.Int64})
    df = pl.DataFrame(
        {
            "s": pl.Series(
                [{"a": None, "b": 1}, {"a": None, "b": 2}, {"a": 5, "b": 5}], dtype=st
            )
        }
    )
    df[:2].write_delta(tmp_path)
    df[2:].write_delta(tmp_path, mode="append")

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = pl.scan_delta(tmp_path).filter(pl.col("s").is_not_null()).collect()
    err = capfd.readouterr().err

    # Every struct is non-null, so all rows are kept; the first file (`a` all-null)
    # must not be pruned.
    assert_frame_equal(out, df, check_row_order=False)
    assert "skipping 1 / 2 files" not in err


@pytest.mark.write_disk
def test_scan_delta_resolves_heavy_footers(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from functools import partial

    import polars.io.delta._scan_resolver as delta_resolver

    plmonkeypatch.setenv("POLARS_DELTA_PLANNER", "deltalake")

    plmonkeypatch.setattr(
        delta_resolver,
        "scan_parquet",
        partial(scan_parquet, _resolve_heavy_sources=4),
    )

    # Heavy-footer resolution needs the file sizes recorded in the Delta log.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    properties = WriterProperties(max_row_group_size=500)

    for n in [20, 4000, 20]:
        pl.DataFrame({"x": range(n)}).write_delta(
            tmp_path,
            mode="append",
            delta_write_options={"writer_properties": properties},
        )

    dataset = new_pl_delta_dataset(DeltaTable(tmp_path))
    capfd.readouterr()
    lf = dataset.lazy()

    retained = lf._ldf._retained_parquet_footers()
    # Retain source 0 and the heavy file, regardless of planner file ordering.
    assert len(retained) == 1
    first_index, first_groups = retained[0][0]
    assert first_index == 0
    assert first_groups in (1, 8)
    assert sorted(rg for _, rg in retained[0]) == ([8] if first_groups == 8 else [1, 8])

    assert lf.collect().height == 4040
    pinned = 0 if first_groups == 8 else 1
    assert (
        f"parquet resolve: pinned {pinned} / {pinned} heavy sources"
        in capfd.readouterr().err
    )


@pytest.mark.parametrize(
    "partition_values", [["0", "1"], ["a b", "100%"], ["\u00e4/\u00f6", "z"]]
)
@pytest.mark.write_disk
def test_scan_delta_source_sizes_match_the_file_list(
    tmp_path: Path, partition_values: list[str]
) -> None:
    # Match sizes against full paths, including encoded partition directories.
    for value in partition_values:
        pl.DataFrame({"p": [value] * 10, "x": range(10)}).write_delta(
            tmp_path, mode="append", delta_write_options={"partition_by": "p"}
        )

    table = DeltaTable(tmp_path)
    uris = table.file_uris()

    assert _source_sizes(
        uris, _table_root(table.table_uri), table._table.get_add_file_sizes()
    ) == [Path(uri).stat().st_size for uri in uris]


def test_delta_table_root_normalisation() -> None:
    # Match local file URIs and Polars' lakefs-to-s3 rewrite.
    assert _table_root("file:///private/var/t") == "/private/var/t/"
    assert _table_root("file:///C:/Users/t") == "C:/Users/t/"
    assert _table_root("s3://bucket/t/") == "s3://bucket/t/"
    assert _table_root("lakefs://repo/main/t") == "s3://repo/main/t/"


def test_delta_source_sizes_lookup() -> None:
    root = "/t/"
    # Files in different partitions can share a name; key on the relative path.
    sizes = {"p=0/part.parquet": 100, "p=1/part.parquet": 200}
    paths = ["/t/p=0/part.parquet", "/t/p=1/part.parquet"]

    assert _source_sizes(paths, root, sizes) == [100, 200]
    # The result follows the scan order, not the dict order.
    assert _source_sizes(paths[::-1], root, dict(reversed(sizes.items()))) == [200, 100]

    # A nested directory that repeats the table name is a distinct key.
    nested = {"part.parquet": 100, "t/part.parquet": 200}
    assert _source_sizes(["/t/part.parquet", "/t/t/part.parquet"], root, nested) == [
        100,
        200,
    ]

    # Unknown paths disable size hints.
    assert _source_sizes(["/other/part.parquet"], root, sizes) is None
    assert _source_sizes(["/t/p=2/part.parquet"], root, sizes) is None


@pytest.mark.write_disk
def test_scan_delta_resolves_heavy_footers_with_encoded_partition_values(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from functools import partial

    import polars.io.delta._scan_resolver as delta_resolver

    plmonkeypatch.setenv("POLARS_DELTA_PLANNER", "deltalake")

    plmonkeypatch.setattr(
        delta_resolver,
        "scan_parquet",
        partial(scan_parquet, _resolve_heavy_sources=4),
    )

    # Encoded partition values must not disable sizes or heavy-footer resolution.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    properties = WriterProperties(max_row_group_size=500)

    for value, n in [("a b", 20), ("100%", 4000), ("\u00e4/\u00f6", 20)]:
        pl.DataFrame({"p": [value] * n, "x": range(n)}).write_delta(
            tmp_path,
            mode="append",
            delta_write_options={
                "partition_by": "p",
                "writer_properties": properties,
            },
        )

    dataset = new_pl_delta_dataset(DeltaTable(tmp_path))
    capfd.readouterr()
    lf = dataset.lazy()

    retained = lf._ldf._retained_parquet_footers()
    # Retain source 0 and the heavy file, regardless of planner file ordering.
    assert len(retained) == 1
    first_index, first_groups = retained[0][0]
    assert first_index == 0
    assert first_groups in (1, 8)
    assert sorted(rg for _, rg in retained[0]) == ([8] if first_groups == 8 else [1, 8])

    assert lf.collect().height == 4040
    pinned = 0 if first_groups == 8 else 1
    assert (
        f"parquet resolve: pinned {pinned} / {pinned} heavy sources"
        in capfd.readouterr().err
    )


@pytest.mark.slow
@pytest.mark.write_disk
def test_scan_delta_resolves_heavy_footers_on_object_store(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from functools import partial

    import polars.io.delta._scan_resolver as delta_resolver

    plmonkeypatch.setenv("POLARS_DELTA_PLANNER", "deltalake")

    plmonkeypatch.setattr(
        delta_resolver,
        "scan_parquet",
        partial(scan_parquet, _resolve_heavy_sources=4),
    )

    # Match encoded paths to sizes, then read source 0 and the heavy-file footer.
    import threading

    from tests.unit.io.cloud.conftest import CountingS3

    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    s3.client.create_bucket(Bucket="bucket")

    try:
        storage_options = {
            "AWS_ACCESS_KEY_ID": s3.storage_options["aws_access_key_id"],
            "AWS_SECRET_ACCESS_KEY": s3.storage_options["aws_secret_access_key"],
            "AWS_REGION": s3.storage_options["aws_region"],
            "AWS_ENDPOINT_URL": s3.endpoint,
            "AWS_ALLOW_HTTP": "true",
            # Allow commits to mock S3 without an external lock.
            "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
        }
        properties = WriterProperties(max_row_group_size=500)

        for value, n in [("a b", 20), ("100%", 4000), ("ä/ö", 20)]:
            pl.DataFrame({"p": [value] * n, "x": range(n)}).write_delta(
                "s3://bucket/t",
                mode="append",
                storage_options=storage_options,
                delta_write_options={
                    "partition_by": "p",
                    "writer_properties": properties,
                },
            )

        table = DeltaTable("s3://bucket/t", storage_options=storage_options)
        uris = table.file_uris()

        # Encoded partition values must not break the size pairing.
        sizes = _source_sizes(
            uris, _table_root(table.table_uri), table._table.get_add_file_sizes()
        )
        assert sizes is not None
        assert len(sizes) == len(uris)

        dataset = DeltaScanResolver(
            table_=NoPickleOption(table),
            table_uri_=None,
            version=None,
            storage_options=storage_options,  # type: ignore[arg-type]
            credential_provider_builder=None,
            delta_table_options=None,
            use_pyarrow=False,
            pyarrow_options=None,
        )

        plmonkeypatch.setenv("POLARS_VERBOSE", "1")
        capfd.readouterr()

        lf = dataset.lazy()

        mark = len(s3.log)
        retained = lf._ldf._retained_parquet_footers()
        planning = s3.since(mark)

        heavy = sizes.index(max(sizes))
        assert retained == [[(0, 1), (heavy, 8)]]
        assert "parquet resolve: pinned 1 / 1 heavy sources" in capfd.readouterr().err

        # Only the retained sources had their footers fetched.
        assert s3.range_get_keys(planning) == {
            uris[i].removeprefix("s3://") for i, _ in retained[0]
        }
    finally:
        s3.server.shutdown()


def join_structure(lf: pl.LazyFrame) -> list[str]:
    return [
        line.strip()
        for line in lf.explain().splitlines()
        if "JOIN" in line or "ON:" in line or "SCAN" in line
    ]


def small_build_side(*keys: int, key: str) -> pl.LazyFrame:
    # Only a filtered build side publishes a runtime filter.
    lf = pl.LazyFrame({key: list(keys), "e": list(range(len(keys)))})
    return lf.filter(pl.col("e") >= 0)


@pytest.mark.write_disk
def test_scan_delta_join_order_matches_parquet(tmp_path: Path) -> None:
    n = 1000
    frames = {
        "a": pl.DataFrame({"k": [i % 5 for i in range(n)], "x": range(n)}),
        "b": pl.DataFrame({"k": [i % 5 for i in range(n)], "y": range(n)}),
        "d": pl.DataFrame({"k": range(5), "flag": [True] + 4 * [False]}),
    }
    for name, df in frames.items():
        df.write_delta(tmp_path / name)
    files = {name: DeltaTable(tmp_path / name).file_uris() for name in frames}

    def query(scan: Callable[[str], pl.LazyFrame]) -> pl.LazyFrame:
        return (
            scan("a")
            .join(scan("b"), on="k")
            .join(scan("d").filter(pl.col("flag")), on="k")
        )

    delta = query(lambda name: pl.scan_delta(tmp_path / name))
    parquet = query(lambda name: pl.scan_parquet(files[name]))

    assert_frame_equal(delta.collect().sort("*"), parquet.collect().sort("*"))
    structure = join_structure(delta)
    if structure != join_structure(parquet):
        pytest.xfail("Delta resolver statistics become available after join ordering")

    # The filtered dimension is joined before the other fact.
    plan = "\n".join(structure)
    assert plan.index(files["d"][0]) < plan.index(files["b"][0])


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.write_disk
def test_scan_delta_runtime_join_filter(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    renamed: bool,
) -> None:
    # The scans here are far smaller than the probe sides worth filtering by default.
    plmonkeypatch.setenv("POLARS_JOIN_RUNTIME_FILTER_MIN_PROBE_ROWS", "0")
    # One file per append.
    for i in range(3):
        pl.DataFrame({"x": range(100 * i, 100 * i + 100), "f": range(100)}).write_delta(
            tmp_path, mode="append"
        )

    # The table statistics hold `f`. They only hold `x` if the join key has its name.
    probe = pl.scan_delta(tmp_path).filter(pl.col("f") >= 0)
    if not renamed:
        # The statistics of `x` skip a file.
        probe = probe.filter(pl.col("x") < 200)
    key = "k" if renamed else "x"
    q = probe.rename({"x": key}).join(small_build_side(150, 160, key=key), on=key)
    assert 'col("x").dynamic_predicate()' in q.explain(engine="streaming")

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    # The reader prunes row groups only with its capabilities.
    plmonkeypatch.setenv("POLARS_FORCE_EMPTY_READER_CAPABILITIES", "0")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err

    skipped = 0 if renamed else 1
    assert f"allows skipping {skipped} / 3 files" in err
    assert err.count("reading 0 / 1 row groups") == 2 - skipped
    assert err.count("reading 1 / 1 row groups") == 1
    assert_frame_equal(
        out,
        pl.DataFrame({key: [150, 160], "f": [50, 60], "e": [0, 1]}),
        check_row_order=False,
    )


@pytest.mark.write_disk
def test_scan_delta_runtime_join_filter_with_partition_predicate(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The scans here are far smaller than the probe sides worth filtering by default.
    plmonkeypatch.setenv("POLARS_JOIN_RUNTIME_FILTER_MIN_PROBE_ROWS", "0")
    for p in range(3):
        pl.DataFrame({"x": range(100 * p, 100 * p + 100), "p": p}).write_delta(
            tmp_path, mode="append", delta_write_options={"partition_by": "p"}
        )

    q = (
        pl.scan_delta(tmp_path)
        .filter(pl.col("p") >= 1)
        .join(small_build_side(150, 160, key="x"), on="x")
        .select("x", "e")
    )

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    # The reader prunes row groups only with its capabilities.
    plmonkeypatch.setenv("POLARS_FORCE_EMPTY_READER_CAPABILITIES", "0")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err

    # The partition predicate skips a file, and the reader still gets the runtime
    # filter.
    assert "allows skipping 1 / 3 files" in err
    assert err.count("reading 0 / 1 row groups") == 1
    assert err.count("reading 1 / 1 row groups") == 1
    assert_frame_equal(
        out, pl.DataFrame({"x": [150, 160], "e": [0, 1]}), check_row_order=False
    )


@pytest.mark.write_disk
def test_scan_delta_partition_values_from_log(tmp_path: Path) -> None:
    root = tmp_path / "version=1"
    df = pl.DataFrame(
        {
            "x": [0, 1, 2],
            "p": [0, 1, None],
            "s": ["a=b/c", "sp ace", None],
            "d": [date(2020, 1, 1), date(2020, 1, 2), None],
            "t": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 2, tzinfo=UTC),
                None,
            ],
        }
    )
    df.write_delta(root, delta_write_options={"partition_by": ["p", "s", "d", "t"]})

    lf = pl.scan_delta(root)
    assert_frame_equal(lf.collect(), df, check_row_order=False)
    assert_frame_equal(lf.filter(pl.col("s") == "a=b/c").collect(), df[:1])
    assert_frame_equal(lf.filter(pl.col("t").is_null()).collect(), df[2:])


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_scan_delta_partition_filter_projects_data_column(
    tmp_path: Path, engine: Literal["in-memory", "streaming"]
) -> None:
    # A predicate fully consumed by partition pruning must not be bound to the
    # projected schema, which no longer holds its column.
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "p": ["A", "B", "B", None, "a=b/c", "a=b/c"],
            "y": [10, 20, 30, 40, 50, 60],
        }
    )
    df.write_delta(tmp_path, delta_write_options={"partition_by": ["p"]})
    lf = pl.scan_delta(tmp_path)

    def xs(q: pl.LazyFrame) -> list[int]:
        return sorted(q.select("x").collect(engine=engine)["x"].to_list())

    b = lf.filter(pl.col("p") == "B")
    assert xs(b) == [2, 3]
    assert b.select(pl.len()).collect(engine=engine).item() == 2
    assert b.select(pl.col("y").sum()).collect(engine=engine).item() == 50
    assert xs(lf.filter(pl.col("p").is_null())) == [4]
    assert xs(lf.filter(pl.col("p") == "a=b/c")) == [5, 6]
    # Mixed predicates still read and filter the data column.
    assert xs(lf.filter((pl.col("p") == "B") & (pl.col("y") > 20))) == [3]
    assert xs(lf.filter((pl.col("p") == "B") | (pl.col("y") == 50))) == [2, 3, 5]


@pytest.mark.write_disk
@pytest.mark.parametrize("checkpoint", [False, True])
def test_scan_delta_binary_partition_values_round_trip(
    tmp_path: Path, checkpoint: bool
) -> None:
    values = [b"hello", b"a/b%z", b"\x00\xff", rb"\u0061", None]
    df = pl.DataFrame({"id": range(len(values)), "part": values})
    write_deltalake(tmp_path, df.to_arrow(), partition_by=["part"])
    if checkpoint:
        DeltaTable(tmp_path).create_checkpoint()

    lf = pl.scan_delta(tmp_path)
    assert_frame_equal(lf.collect(), df, check_row_order=False)
    for value in values[:4]:
        out = lf.filter(pl.col("part") == pl.lit(value)).collect()
        assert out["part"].to_list() == [value]
    assert lf.filter(pl.col("part").is_null()).collect()["id"].to_list() == [4]


@pytest.mark.write_disk
def test_scan_delta_aggregates_do_not_load_column_statistics(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch
) -> None:
    # Delta column statistics are loose: aggregates neither use nor load them.
    # Physical row totals are still loaded for count and runtime planning.
    pl.DataFrame({"a": [1, 2]}).write_delta(tmp_path / "t")
    pl.DataFrame({"a": [3]}).write_delta(tmp_path / "t", mode="append")
    DeltaTable.create(tmp_path / "empty", pa.schema([("a", pa.int64())]))
    calls = []
    import polars.io.delta._scan_resolver as delta_resolver

    extract = _extract_table_statistics_from_delta_add_actions

    def count(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return extract(*args, **kwargs)

    plmonkeypatch.setattr(
        delta_resolver, "_extract_table_statistics_from_delta_add_actions", count
    )
    for name, expected in [("t", (3, 3)), ("empty", (None, 0))]:
        q = pl.scan_delta(tmp_path / name).select(pl.col("a").max(), pl.len())
        assert q.collect().row(0) == expected
        # A duplicated subplan
        assert pl.concat([q, q]).collect().rows() == [expected, expected]
    assert calls == []


def _planner_or_skip(plmonkeypatch: PlMonkeyPatch, planner: str) -> None:
    if planner == "plugin":
        pytest.importorskip("polars_delta")
    plmonkeypatch.setenv("POLARS_DELTA_PLANNER", planner)


def _set_reader_features(root: Path, features: list[str]) -> None:
    import json

    entry = root / "_delta_log" / f"{0:020}.json"
    lines = []
    for line in entry.read_text().splitlines():
        action = json.loads(line)
        if "protocol" in action:
            action["protocol"] = {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": features,
                "writerFeatures": features,
            }
        lines.append(json.dumps(action))
    entry.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("planner", ["deltalake", "plugin"])
@pytest.mark.parametrize(
    "features",
    [
        ["v2Checkpoint"],
        ["vacuumProtocolCheck"],
        ["vacuumProtocolCheck", "v2Checkpoint", "timestampNtz"],
    ],
)
@pytest.mark.write_disk
def test_scan_delta_log_only_reader_features(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, planner: str, features: list[str]
) -> None:
    # These features change how the log is read and what vacuum checks, not the data.
    _planner_or_skip(plmonkeypatch, planner)
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_delta(tmp_path)
    _set_reader_features(tmp_path, features)
    assert_frame_equal(pl.read_delta(tmp_path), df)
    assert_frame_equal(
        pl.scan_delta(tmp_path).filter(pl.col("a") > 1).collect(), df[1:]
    )


@pytest.mark.parametrize("planner", ["deltalake", "plugin"])
@pytest.mark.write_disk
def test_scan_delta_unsupported_reader_feature_is_named(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, planner: str
) -> None:
    _planner_or_skip(plmonkeypatch, planner)
    pl.DataFrame({"a": [1]}).write_delta(tmp_path)
    _set_reader_features(tmp_path, ["v2Checkpoint", "somethingNew"])
    with pytest.raises(
        (
            DeltaError,
            DeltaProtocolError,
            pl.exceptions.ComputeError,
            NotImplementedError,
        )
    ) as e:
        pl.read_delta(tmp_path)
    assert "somethingNew" in str(e.value)
    assert "v2Checkpoint" not in str(e.value)


@pytest.mark.parametrize("planner", ["deltalake", "plugin"])
@pytest.mark.write_disk
def test_scan_delta_casts_files_written_by_other_writers(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, planner: str
) -> None:
    # A data file whose Parquet types differ from the table's (another timezone, a
    # categorical or a wider float) reads as the table's schema, as with deltalake.
    _planner_or_skip(plmonkeypatch, planner)
    ts = datetime(2024, 1, 1, 12, tzinfo=UTC)
    expected = pl.DataFrame(
        {
            "ts": pl.Series([ts, ts]).dt.convert_time_zone("UTC"),
            "c": ["a", "b"],
            "f": pl.Series([1.5, 2.5], dtype=pl.Float32),
        }
    )
    expected.write_delta(tmp_path)
    file = next(tmp_path.rglob("*.parquet"))
    for other in [
        {"ts": pl.col("ts").dt.convert_time_zone("Europe/Amsterdam")},
        {"c": pl.col("c").cast(pl.Categorical)},
        {"f": pl.col("f").cast(pl.Float64)},
    ]:
        expected.with_columns(**other).write_parquet(file)
        _resize_delta_add_actions(tmp_path)
        assert_frame_equal(pl.scan_delta(tmp_path).collect(), expected)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    "schema",
    [
        pa.schema([("a", pa.int32()), ("b", pa.int64())]),
        pa.schema([("b", pa.int64()), ("a", pa.int64())]),
        pa.schema([("a", pa.int64())]),
        pa.schema([("a", pa.int64()), ("b", pa.int64()), ("extra", pa.int32())]),
    ],
    ids=["cast", "reorder", "subset", "extra"],
)
def test_scan_delta_pyarrow_schema_override_matches_data(
    tmp_path: Path, schema: pa.Schema
) -> None:
    write_deltalake(tmp_path, pa.table({"a": [11, 22], "b": [101, 202]}))
    dataset = DeltaTable(tmp_path).to_pyarrow_dataset(schema=schema)
    expected = pl.scan_pyarrow_dataset(dataset).collect()

    lf = pl.scan_delta(tmp_path, use_pyarrow=True, pyarrow_options={"schema": schema})
    assert lf.collect_schema() == expected.schema
    assert_frame_equal(lf.collect(), expected)


@pytest.mark.write_disk
def test_scan_delta_supplied_table_after_log_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.io.delta import _plugin

    write_deltalake(tmp_path, pa.table({"x": [0]}))
    held = DeltaTable(tmp_path)
    write_deltalake(tmp_path, pa.table({"x": [1]}), mode="append")
    DeltaTable(tmp_path).create_checkpoint()
    (tmp_path / "_delta_log" / f"{0:020}.json").unlink()
    assert held.version() == 0

    if os.environ.get("POLARS_DELTA_PLANNER") == "plugin":
        # The native planner is required: it cannot reopen the removed version.
        with pytest.raises(ValueError, match="No table version found"):
            pl.scan_delta(held).collect()
        return

    assert pl.scan_delta(held).collect().rows() == [(0,)]
    assert pl.read_delta(held).rows() == [(0,)]

    # The fallback is remembered: later collects do not retry native planning.
    if not os.environ.get("POLARS_DELTA_PLANNER"):
        calls = []
        original = _plugin.plugin_scan

        def counting_plugin_scan(*a: Any, **k: Any) -> Any:
            calls.append(1)
            return original(*a, **k)

        monkeypatch.setattr(_plugin, "plugin_scan", counting_plugin_scan)
        dataset = new_pl_delta_dataset(held)
        assert dataset._plugin_scan(schema_only=True) is None
        assert dataset._plugin_scan(schema_only=True) is None
        assert len(calls) == 1


@pytest.mark.write_disk
def test_scan_delta_supplied_table_serialization_keeps_loaded_version(
    tmp_path: Path,
) -> None:
    write_deltalake(tmp_path, pa.table({"x": [0]}))
    held = DeltaTable(tmp_path)
    fresh = pl.scan_delta(held)
    collected = pl.scan_delta(held)
    assert collected.collect().rows() == [(0,)]
    write_deltalake(tmp_path, pa.table({"x": [1]}), mode="append")
    held.update_incremental()
    write_deltalake(tmp_path, pa.table({"x": [2]}), mode="append")
    assert held.version() == 1

    for scan in (fresh, collected):
        restored = pickle.loads(pickle.dumps(scan))
        assert sorted(restored.collect().get_column("x")) == [0, 1]


def test_delta_ambiguous_root_never_falls_back_to_other_interpretation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polars._plr as plr
    from polars.io.delta import _plugin

    def unsupported(*_: Any, **__: Any) -> None:
        msg = "unsupported table feature"
        raise NotImplementedError(msg)

    monkeypatch.setattr(plr, "_delta_plugin_scan", unsupported, raising=False)
    monkeypatch.setattr(_plugin, "_plugin_capsule", lambda _: object())
    monkeypatch.setattr(_plugin, "use_plugin_planner", lambda: True)
    monkeypatch.delenv("POLARS_DELTA_PLANNER", raising=False)
    monkeypatch.setattr(
        "polars.io.delta._scan_resolver._get_delta_lake_table",
        lambda **_: pytest.fail("must not reopen the path through deltalake"),
        raising=False,
    )

    dataset = new_pl_delta_dataset("s3://bucket/literal%20space")
    # Native planning opened the raw key (as a restored scan would): no fallback.
    dataset._root_planner = "plugin"
    with pytest.raises(pl.exceptions.ComputeError, match="ambiguous"):
        dataset._plugin_scan(schema_only=True)

    # The guard is recorded before planning, so a failure keeps the identity.
    dataset = new_pl_delta_dataset("s3://bucket/literal%20space")
    with pytest.raises(NotImplementedError):
        _plugin.plugin_scan(dataset, schema_only=True)
    assert dataset._root_planner == "plugin"


def test_delta_auto_planner_uses_deltalake_for_ambiguous_caller_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polars.io.delta import _plugin

    calls = []
    monkeypatch.setattr(_plugin, "use_plugin_planner", lambda: True)

    def fake_plugin_scan(*_: Any, **__: Any) -> Any:
        calls.append(1)
        return None, 0

    monkeypatch.setattr(_plugin, "plugin_scan", fake_plugin_scan)
    uri = "s3://bucket/literal%20space"

    monkeypatch.delenv("POLARS_DELTA_PLANNER", raising=False)
    assert new_pl_delta_dataset(uri)._plugin_scan(schema_only=True) is None
    assert calls == []

    # Explicit `plugin` keeps the raw native root.
    monkeypatch.setenv("POLARS_DELTA_PLANNER", "plugin")
    assert new_pl_delta_dataset(uri)._plugin_scan(schema_only=True) == (None, 0)
    assert calls == [1]


def test_delta_ambiguous_root_without_deltalake_explains_the_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from polars.io.delta import _plugin

    monkeypatch.setattr(_plugin, "use_plugin_planner", lambda: True)
    monkeypatch.delenv("POLARS_DELTA_PLANNER", raising=False)
    monkeypatch.setitem(sys.modules, "deltalake", None)
    dataset = new_pl_delta_dataset("s3://bucket/literal%20space")
    with pytest.raises(ImportError, match=r"ambiguous.*deltalake.*PLANNER=plugin"):
        dataset._plugin_scan(schema_only=True)


@pytest.mark.parametrize("table_id", [None, ""])
def test_delta_expected_table_id_ignores_a_table_without_id(
    table_id: str | None,
) -> None:
    class Metadata:
        id = table_id

    class Table:
        def metadata(self) -> Metadata:
            return Metadata()

    dataset = new_pl_delta_dataset("s3://bucket/t")
    dataset.table_.set(Table())  # type: ignore[arg-type]
    assert dataset._expected_table_id() is None
    Metadata.id = "abc"  # type: ignore[assignment]
    assert dataset._expected_table_id() == "abc"


@pytest.mark.write_disk
def test_scan_delta_pyarrow_old_state_with_schema_override(tmp_path: Path) -> None:
    write_deltalake(tmp_path, pa.table({"a": [11, 22], "b": [101, 202]}))
    schema = pa.schema([("b", pa.int64()), ("a", pa.int32())])
    expected = pl.scan_pyarrow_dataset(
        DeltaTable(tmp_path).to_pyarrow_dataset(schema=schema)
    ).collect()
    dataset = DeltaScanResolver(
        table_=NoPickleOption(None),
        table_uri_=str(tmp_path),
        version=None,
        storage_options=None,
        credential_provider_builder=None,
        delta_table_options=None,
        use_pyarrow=True,
        pyarrow_options={"schema": schema},
    )
    # Older pickles kept the Delta schema in `_schema` and had no `_table_schema`.
    state = dataset.__getstate__()
    state["_schema"] = pl.Schema(DeltaTable(tmp_path).schema())
    state.pop("_table_schema", None)
    dataset.__setstate__(state)
    restored = pickle.loads(pickle.dumps(dataset))

    frame, _ = restored._resolve_scan(
        existing_resolved_version_key=None, projection=None, filter_columns=None
    )
    assert frame is not None
    assert_frame_equal(frame.collect(), expected)


@pytest.mark.write_disk
def test_scan_delta_pyarrow_builds_one_dataset_per_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_deltalake(tmp_path, pa.table({"a": [1]}))
    calls = []
    original = DeltaTable.to_pyarrow_dataset

    def counting_to_pyarrow_dataset(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return original(*a, **k)

    monkeypatch.setattr(DeltaTable, "to_pyarrow_dataset", counting_to_pyarrow_dataset)
    lf = pl.scan_delta(tmp_path, use_pyarrow=True)
    assert lf.collect_schema() == {"a": pl.Int64}
    assert lf.collect().rows() == [(1,)]
    assert len(calls) == 1
    # A newer version builds another one.
    write_deltalake(tmp_path, pa.table({"a": [2]}), mode="append")
    assert sorted(pl.scan_delta(tmp_path, use_pyarrow=True).collect()["a"]) == [1, 2]
