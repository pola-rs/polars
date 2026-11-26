from __future__ import annotations

import importlib
import json
import struct
import zlib
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def plugin_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("polars_delta")
    monkeypatch.setenv("POLARS_DELTA_PLANNER", "plugin")


def _table(
    path: Path,
    fields: list[dict[str, Any]],
    data: list[pl.DataFrame],
    *,
    mode: str = "none",
    features: list[str] | None = None,
    partition_columns: list[str] | None = None,
    partitions: list[dict[str, str | None]] | None = None,
) -> Path:
    path.mkdir()
    log = path / "_delta_log"
    log.mkdir()
    configuration = {"delta.columnMapping.mode": mode} if mode != "none" else {}
    reader_features = features or (["columnMapping"] if mode != "none" else [])
    actions: list[dict[str, Any]] = [
        {
            "protocol": {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": reader_features,
                "writerFeatures": reader_features,
            }
        },
        {
            "metaData": {
                "id": "delta-plugin-test",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
                "partitionColumns": partition_columns or [],
                "configuration": configuration,
                "createdTime": 0,
            }
        },
    ]
    for i, frame in enumerate(data):
        file = path / f"part-{i}.parquet"
        frame.write_parquet(file)
        actions.append(
            {
                "add": {
                    "path": file.name,
                    "size": file.stat().st_size,
                    "partitionValues": partitions[i] if partitions else {},
                    "modificationTime": 0,
                    "dataChange": True,
                    "stats": json.dumps({"numRecords": len(frame)}),
                }
            }
        )
    (log / "00000000000000000000.json").write_text(
        "\n".join(json.dumps(a) for a in actions)
    )
    return path


def _field(name: str, dtype: Any, **metadata: Any) -> dict[str, Any]:
    return {"name": name, "type": dtype, "nullable": True, "metadata": metadata}


def test_delta_plugin_without_deltalake_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.io.delta._scan_resolver import DeltaScanResolver

    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1, 2]})])

    def forbid_table(self: DeltaScanResolver) -> None:
        pytest.fail("native Delta planning initialized deltalake")

    monkeypatch.setattr(DeltaScanResolver, "table", forbid_table)
    scan = pl.scan_delta(path)
    assert scan.collect_schema() == {"a": pl.Int64}
    assert_frame_equal(scan.collect(), pl.DataFrame({"a": [1, 2]}))


def test_delta_plugin_schema_pins_snapshot(tmp_path: Path) -> None:
    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    scan = pl.scan_delta(path)
    assert scan.collect_schema() == {"a": pl.Int64}
    file = path / "next.parquet"
    pl.DataFrame({"b": [2]}).write_parquet(file)
    actions = [
        {
            "metaData": {
                "id": "delta-plugin-test",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps(
                    {"type": "struct", "fields": [_field("b", "long")]}
                ),
                "partitionColumns": [],
                "configuration": {},
            }
        },
        {
            "remove": {
                "path": "part-0.parquet",
                "deletionTimestamp": 1,
                "dataChange": True,
            }
        },
        {
            "add": {
                "path": file.name,
                "size": file.stat().st_size,
                "partitionValues": {},
                "modificationTime": 1,
                "dataChange": True,
            }
        },
    ]
    (path / "_delta_log/00000000000000000001.json").write_text(
        "\n".join(json.dumps(a) for a in actions)
    )
    assert_frame_equal(scan.collect(), pl.DataFrame({"a": [1]}))
    assert_frame_equal(pl.scan_delta(path).collect(), pl.DataFrame({"b": [2]}))


def test_delta_plugin_relative_path_pins_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for value in (1, 2):
        parent = tmp_path / str(value)
        parent.mkdir()
        _table(parent / "t", [_field("a", "long")], [pl.DataFrame({"a": [value]})])

    monkeypatch.chdir(tmp_path / "1")
    scan = pl.scan_delta("t")
    assert scan.collect_schema() == {"a": pl.Int64}
    monkeypatch.chdir(tmp_path / "2")
    assert_frame_equal(scan.collect(), pl.DataFrame({"a": [1]}))


@pytest.mark.parametrize("mode", ["name", "id"])
def test_delta_plugin_column_mapping_with_conflicting_ids(
    tmp_path: Path, mode: str
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [
        _field(
            "renamed",
            "long",
            **{
                "delta.columnMapping.id": 1,
                "delta.columnMapping.physicalName": "physical",
            },
        )
    ]
    path = _table(tmp_path / "t", fields, [], mode=mode)
    # Name mode ignores conflicting IDs; ID mode ignores conflicting names.
    name, field_id = ("physical", "99") if mode == "name" else ("other", "1")
    schema = pa.schema(
        [pa.field(name, pa.int64(), metadata={b"PARQUET:field_id": field_id.encode()})]
    )
    file = path / "data.parquet"
    pq.write_table(pa.Table.from_arrays([pa.array([7])], schema=schema), file)
    with (path / "_delta_log/00000000000000000000.json").open("a") as f:
        f.write(
            "\n"
            + json.dumps(
                {
                    "add": {
                        "path": file.name,
                        "size": file.stat().st_size,
                        "partitionValues": {},
                        "dataChange": True,
                    }
                }
            )
        )
    assert_frame_equal(pl.scan_delta(path).collect(), pl.DataFrame({"renamed": [7]}))


@pytest.mark.parametrize("nested", [False, True])
def test_delta_plugin_dropped_readded_name(tmp_path: Path, nested: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    field = _field(
        "a",
        "long",
        **{"delta.columnMapping.id": 2, "delta.columnMapping.physicalName": "new"},
    )
    physical = pa.field("old", pa.int64(), metadata={b"PARQUET:field_id": b"2"})
    expected = pl.DataFrame({"a": [None]}, schema={"a": pl.Int64})
    values: list[Any]
    if nested:
        field = _field(
            "s",
            {"type": "struct", "fields": [field]},
            **{
                "delta.columnMapping.id": 1,
                "delta.columnMapping.physicalName": "physical_s",
            },
        )
        physical = pa.field(
            "physical_s", pa.struct([physical]), metadata={b"PARQUET:field_id": b"99"}
        )
        values = [{"old": 1}]
        expected = expected.select(pl.struct(pl.all()).alias("s"))
    else:
        values = [1]
    path = _table(tmp_path / "t", [field], [], mode="name")
    file = path / "data.parquet"
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array(values, type=physical.type)], schema=pa.schema([physical])
        ),
        file,
    )
    with (path / "_delta_log/00000000000000000000.json").open("a") as f:
        f.write(
            "\n"
            + json.dumps(
                {
                    "add": {
                        "path": file.name,
                        "size": file.stat().st_size,
                        "partitionValues": {},
                        "dataChange": True,
                    }
                }
            )
        )
    assert_frame_equal(pl.scan_delta(path).collect(), expected)


@pytest.mark.parametrize(
    ("incoming", "target", "delta_type", "from_type", "to_type"),
    [
        (pl.Int8, pl.Int16, "short", "byte", "short"),
        (pl.Int16, pl.Int32, "integer", "short", "integer"),
        (pl.Int32, pl.Int64, "long", "integer", "long"),
        (pl.Float32, pl.Float64, "double", "float", "double"),
        (pl.Int8, pl.Float64, "double", "byte", "double"),
        (pl.Int16, pl.Float64, "double", "short", "double"),
        (pl.Int32, pl.Float64, "double", "integer", "double"),
        (pl.Int8, pl.Decimal(4, 1), "decimal(4,1)", "byte", "decimal(4,1)"),
        (pl.Int16, pl.Decimal(6, 1), "decimal(6,1)", "short", "decimal(6,1)"),
        (pl.Int64, pl.Decimal(21, 1), "decimal(21,1)", "long", "decimal(21,1)"),
        (pl.Date, pl.Datetime("us"), "timestamp_ntz", "date", "timestamp_ntz"),
        (
            pl.Decimal(10, 2),
            pl.Decimal(14, 4),
            "decimal(14,4)",
            "decimal(10,2)",
            "decimal(14,4)",
        ),
        (pl.Int32, pl.Decimal(12, 2), "decimal(12,2)", "integer", "decimal(12,2)"),
    ],
)
def test_delta_plugin_type_widening(
    tmp_path: Path,
    incoming: pl.DataType,
    target: pl.DataType,
    delta_type: str,
    from_type: str,
    to_type: str,
) -> None:
    from datetime import date
    from decimal import Decimal

    value: Any = (
        date(2020, 1, 2)
        if incoming == pl.Date
        else Decimal("12.34")
        if incoming.is_decimal()
        else 12
    )
    frame = pl.DataFrame({"a": [value]}, schema={"a": incoming})
    field = _field(
        "a",
        delta_type,
        **{"delta.typeChanges": [{"fromType": from_type, "toType": to_type}]},
    )
    path = _table(
        tmp_path / "t", [field], [frame], features=["typeWidening", "timestampNtz"]
    )
    assert_frame_equal(pl.scan_delta(path).collect(), frame.cast({"a": target}))


def test_delta_plugin_widening_flag_visible_to_visitor(tmp_path: Path) -> None:
    # An engine that takes the plan through the node visitor (GPU) must see that the
    # scan allows Delta type widening, or it rejects widened files.
    field = _field(
        "a",
        "long",
        **{"delta.typeChanges": [{"fromType": "integer", "toType": "long"}]},
    )
    path = _table(
        tmp_path / "t",
        [field],
        [pl.DataFrame({"a": [1]}, schema={"a": pl.Int32})],
        features=["typeWidening"],
    )
    import polars._plr as plr
    from polars._utils.wrap import wrap_ldf
    from polars.io.delta._plugin import _plugin_capsule
    from polars.io.delta._utils import _resolve_delta_lake_uri

    # The planner's own resolved scan: `scan_delta` returns an unresolved dataset scan,
    # which the GPU engine cannot take (`IR::Resolver`) before it is resolved.
    resolved, _, _ = plr._delta_plugin_scan(
        _plugin_capsule(plr._IO_PLUGIN_IDS),
        table_uri=_resolve_delta_lake_uri(str(path), strict=False),
        version=None,
        timestamp_ms=None,
        projection=None,
        filter_columns=None,
        statistics_columns=None,
        limit=None,
        use_metadata_statistics=True,
        verbose=False,
        testing_fail=None,
        storage_options=None,
        credential_provider=None,
        metadata_cache=plr.PyIcebergMetadataFileCache(1 << 20),
    )
    node = wrap_ldf(resolved)._ldf.visit().view_current_node()
    policy = node.file_options.cast_columns_policy
    assert policy["delta_type_widening"] is True
    # The policy the deltalake path sets.
    assert policy["float_downcast"] is True
    assert policy["datetime_convert_timezone"] is True
    assert policy["categorical_to_string"] is True


def test_delta_plugin_partition_filter_limit(tmp_path: Path) -> None:
    path = _table(
        tmp_path / "t",
        [_field("a", "long"), _field("p", "integer")],
        [pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [2]})],
        partition_columns=["p"],
        partitions=[{"p": "1"}, {"p": "2"}],
    )
    assert_frame_equal(
        pl.scan_delta(path).filter(pl.col("p") == 2).head(1).collect(),
        pl.DataFrame({"a": [2], "p": [2]}, schema={"a": pl.Int64, "p": pl.Int32}),
    )


@pytest.mark.parametrize("offset", [None, 0, 1])
def test_delta_plugin_persisted_dv_offset(tmp_path: Path, offset: int | None) -> None:
    path = _table(
        tmp_path / "t",
        [_field("a", "long")],
        [pl.DataFrame({"a": [0, 1, 2]})],
        features=["deletionVectors"],
    )
    body = struct.pack("<IQIIIHHIH", 1681511377, 1, 0, 12346, 1, 0, 0, 16, 0)
    dv = path / "dv.bin"
    dv.write_bytes(
        b"\x01"
        + struct.pack(">I", len(body))
        + body
        + struct.pack(">I", zlib.crc32(body))
    )
    descriptor = {
        "storageType": "p",
        "pathOrInlineDv": dv.as_uri(),
        "sizeInBytes": len(body),
        "cardinality": 1,
    }
    if offset is not None:
        descriptor["offset"] = offset
    log = path / "_delta_log/00000000000000000000.json"
    actions = [json.loads(line) for line in log.read_text().splitlines()]
    actions[-1]["add"]["deletionVector"] = descriptor
    log.write_text("\n".join(json.dumps(action) for action in actions))
    scan = pl.scan_delta(path)
    queries = [scan, scan.filter(pl.col("a") >= 1)]
    if offset == 1:
        for query in queries:
            assert_frame_equal(query.collect(), pl.DataFrame({"a": [1, 2]}))
            assert query.select(pl.len()).collect().item() == 2
    else:
        # Absent and explicit zero both denote byte zero, not the first frame
        # after this file's version byte. Kernel-rs currently differs for absence.
        for query in queries:
            with pytest.raises(pl.exceptions.ComputeError, match="offset must follow"):
                query.collect()
        # A data-dependent count also reads and validates the DV.
        with pytest.raises(pl.exceptions.ComputeError, match="offset must follow"):
            queries[1].select(pl.len()).collect()


def test_delta_plugin_log_statistics_prune_deletion_vectors(tmp_path: Path) -> None:
    path = _table(
        tmp_path / "t",
        [_field("a", "long")],
        [pl.DataFrame({"a": [0, 1, 2]}), pl.DataFrame({"a": [10, 11, 12]})],
        features=["deletionVectors"],
    )
    # One portable Roaring deletion at physical row zero.
    body = struct.pack("<IQIIIHHIH", 1681511377, 1, 0, 12346, 1, 0, 0, 16, 0)
    dv = path / "second.bin"
    dv.write_bytes(
        b"\x01"
        + struct.pack(">I", len(body))
        + body
        + struct.pack(">I", zlib.crc32(body))
    )
    log = path / "_delta_log/00000000000000000000.json"
    actions = [json.loads(line) for line in log.read_text().splitlines()]
    for i, action in enumerate(a for a in actions if "add" in a):
        action["add"]["stats"] = json.dumps(
            {
                "numRecords": 3,
                "minValues": {"a": i * 10},
                "maxValues": {"a": i * 10 + 2},
                "nullCount": {"a": 0},
                "tightBounds": False,
            }
        )
        action["add"]["deletionVector"] = {
            "storageType": "p",
            "pathOrInlineDv": (
                (path / "missing.bin").as_uri() if i == 0 else dv.as_uri()
            ),
            "offset": 1,
            "sizeInBytes": len(body),
            "cardinality": 1,
        }
    log.write_text("\n".join(json.dumps(action) for action in actions))

    scan = pl.scan_delta(path)
    selected = scan.filter(pl.col("a") >= 10)
    expected = pl.DataFrame({"a": [11, 12]})
    assert_frame_equal(selected.collect(), expected)
    assert_frame_equal(selected.slice(1, 1).collect(), expected.slice(1, 1))
    assert_frame_equal(selected.tail(1).collect(), expected.tail(1))
    # The descriptor is real: scanning its file must attempt the missing DV.
    with pytest.raises(FileNotFoundError, match=r"missing\.bin"):
        scan.collect()


def test_delta_plugin_io_errors_do_not_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polars.io.delta._scan_resolver import DeltaScanResolver

    monkeypatch.delenv("POLARS_DELTA_PLANNER")
    monkeypatch.setattr(
        DeltaScanResolver, "table", lambda self: pytest.fail("IO error fell back")
    )
    with pytest.raises((FileNotFoundError, ValueError, pl.exceptions.ComputeError)):
        pl.scan_delta(tmp_path / "missing").collect()


@pytest.mark.parametrize("serialization", ["pickle", "lazyframe"])
def test_delta_plugin_serialization_preserves_snapshot(
    tmp_path: Path, serialization: str
) -> None:
    import io
    import pickle

    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    scan = pl.scan_delta(path)
    scan.collect_schema()
    if serialization == "pickle":
        restored = pickle.loads(pickle.dumps(scan))
    else:
        restored = pl.LazyFrame.deserialize(io.BytesIO(scan.serialize()))
    assert_frame_equal(restored.collect(), pl.DataFrame({"a": [1]}))


def test_delta_plugin_deserialization_selects_receiver_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pickle
    import sys

    expected = pl.DataFrame({"a": [1]})
    expected.write_delta(tmp_path)
    monkeypatch.delenv("POLARS_DELTA_PLANNER")
    scan = pl.scan_delta(tmp_path)
    assert scan.collect_schema() == expected.schema
    serialized = pickle.dumps(scan)
    pl.DataFrame({"a": [2]}).write_delta(tmp_path, mode="append")

    monkeypatch.setitem(sys.modules, "polars_delta", None)
    restored = pickle.loads(serialized)
    with pytest.warns(pl.exceptions.PerformanceWarning, match="Install polars-delta"):
        assert_frame_equal(restored.collect(), expected)


@pytest.mark.parametrize("error", [NotImplementedError, pl.exceptions.ComputeError])
@pytest.mark.parametrize("required", [False, True])
def test_delta_plugin_fallback_is_only_for_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
    required: bool,
) -> None:
    from polars.io.delta import _plugin

    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    if not required:
        monkeypatch.delenv("POLARS_DELTA_PLANNER")

    def fail(*args: Any, **kwargs: Any) -> Any:
        msg = "injected planner error"
        raise error(msg)

    monkeypatch.setattr(_plugin, "plugin_scan", fail)
    if error is NotImplementedError and not required:
        assert_frame_equal(pl.scan_delta(path).collect(), pl.DataFrame({"a": [1]}))
    else:
        with pytest.raises(error, match="injected planner error"):
            pl.scan_delta(path).collect()


@pytest.mark.parametrize("required", [False, True])
def test_delta_plugin_missing_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: bool
) -> None:
    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    if not required:
        monkeypatch.delenv("POLARS_DELTA_PLANNER")
    original_import = importlib.import_module

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "polars_delta":
            msg = "No module named 'polars_delta'"
            raise ModuleNotFoundError(msg, name="polars_delta")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    if required:
        with pytest.raises(ModuleNotFoundError, match="polars_delta"):
            pl.scan_delta(path).collect()
    else:
        with pytest.warns(
            pl.exceptions.PerformanceWarning, match="Install polars-delta"
        ):
            assert_frame_equal(pl.scan_delta(path).collect(), pl.DataFrame({"a": [1]}))


@pytest.mark.parametrize("required", [False, True])
def test_delta_plugin_unsupported_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: bool
) -> None:
    import sys
    import types

    old = types.ModuleType("polars_delta")
    old._polars_io_plugin_ids = ["polars.io_plugin.delta.v999"]  # type: ignore[attr-defined]
    old._capsule = lambda _id: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "polars_delta", old)
    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    if required:
        with pytest.raises(pl.exceptions.ComputeError, match="no compatible Delta"):
            pl.scan_delta(path).collect()
    else:
        monkeypatch.delenv("POLARS_DELTA_PLANNER")
        with pytest.warns(
            pl.exceptions.PerformanceWarning, match="upgrade polars-delta"
        ):
            assert_frame_equal(pl.scan_delta(path).collect(), pl.DataFrame({"a": [1]}))


@pytest.mark.parametrize("container", ["struct", "array", "map"])
def test_delta_plugin_nested_type_widening(tmp_path: Path, container: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    change = {"fromType": "byte", "toType": "decimal(4,1)"}
    if container == "struct":
        delta_type: dict[str, Any] = {
            "type": "struct",
            "fields": [
                _field("inner", "decimal(4,1)", **{"delta.typeChanges": [change]})
            ],
        }
        dtype = pa.struct([pa.field("inner", pa.int8())])
        values: list[Any] = [{"inner": 12}]
        target: pl.DataType = pl.Struct({"inner": pl.Decimal(4, 1)})
        metadata: dict[str, Any] = {}
    elif container == "array":
        delta_type = {
            "type": "array",
            "elementType": "decimal(4,1)",
            "containsNull": True,
        }
        dtype = pa.list_(pa.int8())
        values = [[12]]
        target = pl.List(pl.Decimal(4, 1))
        metadata = {"delta.typeChanges": [{**change, "fieldPath": "element"}]}
    else:
        delta_type = {
            "type": "map",
            "keyType": "string",
            "valueType": "decimal(4,1)",
            "valueContainsNull": True,
        }
        dtype = pa.map_(pa.string(), pa.int8())
        values = [[("k", 12)]]
        target = pl.Map(pl.String, pl.Decimal(4, 1))
        metadata = {"delta.typeChanges": [{**change, "fieldPath": "value"}]}
    path = _table(
        tmp_path / "t",
        [_field("a", delta_type, **metadata)],
        [],
        features=["typeWidening"],
    )
    file = path / "data.parquet"
    pq.write_table(pa.table({"a": pa.array(values, type=dtype)}), file)
    with (path / "_delta_log/00000000000000000000.json").open("a") as f:
        f.write(
            "\n"
            + json.dumps(
                {
                    "add": {
                        "path": file.name,
                        "size": file.stat().st_size,
                        "partitionValues": {},
                        "dataChange": True,
                    }
                }
            )
        )
    expected = pl.read_parquet(file).cast({"a": target})
    assert_frame_equal(pl.scan_delta(path).collect(), expected)


@pytest.mark.parametrize(
    ("source", "destination"),
    [
        ("byte", "decimal(3,1)"),
        ("short", "decimal(5,1)"),
        ("long", "double"),
        ("decimal(4,1)", "decimal(5,3)"),
    ],
)
def test_delta_plugin_rejects_lossy_type_change(
    tmp_path: Path, source: str, destination: str
) -> None:
    field = _field(
        "a",
        destination,
        **{"delta.typeChanges": [{"fromType": source, "toType": destination}]},
    )
    path = _table(tmp_path / "t", [field], [], features=["typeWidening"])
    with pytest.raises(
        (NotImplementedError, pl.exceptions.ComputeError),
        match=r"(?i)(widen|type|decimal)",
    ):
        pl.scan_delta(path).collect()


@pytest.mark.parametrize("mode", ["name", "id"])
@pytest.mark.parametrize("nesting", ["none", "struct", "list", "map"])
@pytest.mark.parametrize("reversed_children", [False, True])
def test_delta_plugin_mapped_variant_physical_children(
    tmp_path: Path, mode: str, nesting: str, reversed_children: bool
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    mapping = {
        "delta.columnMapping.id": 1,
        "delta.columnMapping.physicalName": "physical",
    }
    children = [
        pa.field("value", pa.binary(), nullable=False),
        pa.field("metadata", pa.binary(), nullable=False),
    ]
    if reversed_children:
        children.reverse()
    dtype = pa.struct(children)
    raw: Any = "variant"
    value: Any = {"value": b"\x01", "metadata": b"\x02"}
    if nesting == "struct":
        raw = {
            "type": "struct",
            "fields": [
                _field(
                    "inner",
                    "variant",
                    **{
                        "delta.columnMapping.id": 2,
                        "delta.columnMapping.physicalName": "physical_inner",
                    },
                )
            ],
        }
        dtype = pa.struct(
            [pa.field("physical_inner", dtype, metadata={b"PARQUET:field_id": b"2"})]
        )
        value = {"physical_inner": value}
    elif nesting == "list":
        raw = {"type": "array", "elementType": "variant", "containsNull": True}
        dtype = pa.list_(dtype)
        value = [value]
    elif nesting == "map":
        raw = {
            "type": "map",
            "keyType": "string",
            "valueType": "variant",
            "valueContainsNull": True,
        }
        dtype = pa.map_(pa.string(), dtype)
        value = [("k", value)]
    field = _field("a", raw, **mapping)
    path = _table(
        tmp_path / "t",
        [field],
        [],
        mode=mode,
        features=["columnMapping", "variantType"],
    )
    scan = pl.scan_delta(path)
    assert scan.collect_schema()["a"] is not None
    assert scan.collect().height == 0
    physical = pa.field("physical", dtype, metadata={b"PARQUET:field_id": b"1"})
    file = path / "data.parquet"
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array([value], type=dtype)], schema=pa.schema([physical])
        ),
        file,
    )
    with (path / "_delta_log/00000000000000000000.json").open("a") as f:
        f.write(
            "\n"
            + json.dumps(
                {
                    "add": {
                        "path": file.name,
                        "size": file.stat().st_size,
                        "partitionValues": {},
                        "dataChange": True,
                    }
                }
            )
        )
    result = pl.scan_delta(path).collect()
    assert result.height == 1
    row = result["a"].to_list()[0]
    if nesting == "struct":
        row = row["inner"]
    elif nesting == "list":
        row = row[0]
    elif nesting == "map":
        row = row["k"]
    assert row == {"value": b"\x01", "metadata": b"\x02"}


@pytest.mark.parametrize("mode", ["name", "id"])
def test_delta_plugin_ordinary_struct_cannot_spoof_variant(
    tmp_path: Path, mode: str
) -> None:
    fields = [
        _field(
            "value",
            "binary",
            **{"delta.columnMapping.id": 2, "delta.columnMapping.physicalName": "v"},
        ),
        _field(
            "metadata",
            "binary",
            **{"delta.columnMapping.id": 3, "delta.columnMapping.physicalName": "m"},
        ),
    ]
    field = _field(
        "a",
        {"type": "struct", "fields": fields},
        **{
            "delta.columnMapping.id": 1,
            "delta.columnMapping.physicalName": "physical",
            "polars.delta.variant": "true",
        },
    )
    path = _table(tmp_path / "t", [field], [], mode=mode)
    assert pl.scan_delta(path).collect_schema()["a"] == pl.Struct(
        {"value": pl.Binary, "metadata": pl.Binary}
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    dtype = pa.struct(
        [pa.field("value", pa.binary()), pa.field("metadata", pa.binary())]
    )
    physical = pa.field("physical", dtype, metadata={b"PARQUET:field_id": b"1"})
    file = path / "data.parquet"
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array([{"value": b"a", "metadata": b"b"}], type=dtype)],
            schema=pa.schema([physical]),
        ),
        file,
    )
    with (path / "_delta_log/00000000000000000000.json").open("a") as f:
        f.write(
            "\n"
            + json.dumps(
                {
                    "add": {
                        "path": file.name,
                        "size": file.stat().st_size,
                        "partitionValues": {},
                        "dataChange": True,
                    }
                }
            )
        )
    assert pl.scan_delta(path).collect()["a"].to_list() == [
        {"value": None, "metadata": None}
    ]


@pytest.mark.parametrize("mode", ["name", "id"])
@pytest.mark.parametrize(
    "bad_children",
    [
        [("value", "binary")],
        [("value", "binary"), ("metadata", "string")],
        [("value", "binary"), ("value", "binary")],
    ],
)
def test_delta_plugin_mapped_variant_invalid_storage(
    tmp_path: Path, mode: str, bad_children: list[tuple[str, str]]
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    field = _field(
        "a",
        "variant",
        **{"delta.columnMapping.id": 1, "delta.columnMapping.physicalName": "physical"},
    )
    path = _table(
        tmp_path / "t",
        [field],
        [],
        mode=mode,
        features=["columnMapping", "variantType"],
    )
    dtype = pa.struct(
        [
            pa.field(name, pa.binary() if kind == "binary" else pa.string())
            for name, kind in bad_children
        ]
    )
    physical = pa.field("physical", dtype, metadata={b"PARQUET:field_id": b"1"})
    file = path / "data.parquet"
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array([None], type=dtype)], schema=pa.schema([physical])
        ),
        file,
    )
    with (path / "_delta_log/00000000000000000000.json").open("a") as f:
        f.write(
            "\n"
            + json.dumps(
                {
                    "add": {
                        "path": file.name,
                        "size": file.stat().st_size,
                        "partitionValues": {},
                        "dataChange": True,
                    }
                }
            )
        )
    with pytest.raises(pl.exceptions.SchemaError, match=r"(?i)variant"):
        pl.scan_delta(path).collect()


def test_delta_plugin_mutable_table_explicit_version(tmp_path: Path) -> None:
    from deltalake import DeltaTable

    pl.DataFrame({"a": [0]}).write_delta(tmp_path)
    table = DeltaTable(tmp_path)
    pl.DataFrame({"a": [1]}).write_delta(table, mode="append")
    scan = pl.scan_delta(table, version=0)
    assert_frame_equal(scan.collect(), pl.DataFrame({"a": [0]}))
    pl.DataFrame({"a": [2]}).write_delta(table, mode="append")
    assert_frame_equal(scan.collect(), pl.DataFrame({"a": [0]}))


def test_delta_plugin_mutable_table_schema_change(tmp_path: Path) -> None:
    from deltalake import DeltaTable

    pl.DataFrame({"a": [0]}).write_delta(tmp_path)
    table = DeltaTable(tmp_path)
    scan = pl.scan_delta(table)
    assert_frame_equal(scan.collect(), pl.DataFrame({"a": [0]}))
    pl.DataFrame({"b": [1]}).write_delta(
        table, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
    )
    with pytest.raises(pl.exceptions.SchemaError, match="schema changed"):
        scan.collect()
    assert_frame_equal(pl.scan_delta(table).collect(), pl.DataFrame({"b": [1]}))


@pytest.mark.parametrize(
    "change", ["projection", "filter_columns", "predicate", "limit"]
)
def test_delta_plugin_reuses_only_covered_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    import polars._plr as plr
    from polars.io.delta._scan_resolver import DeltaScanResolver

    frame = pl.DataFrame({"a": [1, 2], "b": [2, 1]})
    path = _table(tmp_path / "t", [_field("a", "long"), _field("b", "long")], [frame])
    native = plr._delta_plugin_scan
    resolver = DeltaScanResolver.resolve_lazyframe
    calls: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def capture_native(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return native(*args, **kwargs)

    def capture_request(self: DeltaScanResolver, **kwargs: Any) -> Any:
        requests.append(kwargs)
        return resolver(self, **kwargs)

    monkeypatch.setattr(plr, "_delta_plugin_scan", capture_native)
    monkeypatch.setattr(DeltaScanResolver, "resolve_lazyframe", capture_request)
    scan = pl.scan_delta(path)
    if change == "projection":
        before, after = scan.select("a"), scan.select("b")
    elif change == "filter_columns":
        before, after = scan.filter(pl.col("a") > 0), scan.filter(pl.col("b") > 0)
    elif change == "predicate":
        before, after = scan.filter(pl.col("a") > 0), scan.filter(pl.col("a") > 1)
    else:
        assert change == "limit"
        before, after = scan.head(1), scan.head(2)

    before.explain(optimized=True)
    calls.clear()
    requests.clear()
    before.explain(optimized=True)
    assert not calls
    assert requests
    assert all(r["existing_resolved_version_key"] == "0" for r in requests)

    calls.clear()
    requests.clear()
    after.explain(optimized=True)
    assert calls
    assert any(r["existing_resolved_version_key"] is None for r in requests)


def test_delta_plugin_mutable_table_refreshes_cached_scan(tmp_path: Path) -> None:
    from deltalake import DeltaTable

    pl.DataFrame({"a": [0]}).write_delta(tmp_path)
    table = DeltaTable(tmp_path)
    scan = pl.scan_delta(table)
    assert_frame_equal(scan.collect(), pl.DataFrame({"a": [0]}))
    assert_frame_equal(scan.collect(), pl.DataFrame({"a": [0]}))
    pl.DataFrame({"a": [1]}).write_delta(table, mode="append")
    assert_frame_equal(scan.collect().sort("a"), pl.DataFrame({"a": [0, 1]}))


@pytest.mark.parametrize("selector", ["string", "datetime", "naive_datetime"])
def test_delta_plugin_timestamp_pins_resolved_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    from datetime import UTC, datetime

    from deltalake import DeltaTable

    import polars._plr as plr

    aware = datetime(2100, 1, 1, tzinfo=UTC)
    version: str | datetime = aware
    if selector == "string":
        version = aware.isoformat()
    elif selector == "naive_datetime":
        version = aware.replace(tzinfo=None)
    pl.DataFrame({"a": [0], "b": [10]}).write_delta(tmp_path)
    table = DeltaTable(tmp_path)
    native = plr._delta_plugin_scan
    calls: list[dict[str, Any]] = []

    def capture_native(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return native(*args, **kwargs)

    monkeypatch.setattr(plr, "_delta_plugin_scan", capture_native)
    scan = pl.scan_delta(table, version=version)
    assert_frame_equal(scan.select("a").collect(), pl.DataFrame({"a": [0]}))
    calls.clear()
    assert_frame_equal(scan.select("a").collect(), pl.DataFrame({"a": [0]}))
    assert not calls
    pl.DataFrame({"a": [1], "b": [11]}).write_delta(table, mode="append")
    # A different projection must replan the same resolved snapshot, including v0.
    assert_frame_equal(scan.select("b").collect(), pl.DataFrame({"b": [10]}))
    assert len(calls) == 1
    assert calls[0]["version"] == 0
    assert calls[0]["timestamp_ms"] is None


def test_delta_plugin_time_travel_at_exact_millisecond(tmp_path: Path) -> None:
    import os

    pl.DataFrame({"a": [1]}).write_delta(tmp_path)
    expected = pl.DataFrame({"a": [2]})
    expected.write_delta(tmp_path, mode="overwrite")
    for version in (0, 1):
        entry = tmp_path / "_delta_log" / f"{version:020}.json"
        time_ns = (1_075_593_600_000 + version) * 1_000_000
        os.utime(entry, ns=(time_ns, time_ns))

    scan = pl.scan_delta(tmp_path, version="2004-02-01T00:00:00.001Z")
    assert_frame_equal(scan.collect(), expected)


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_plugin_cloud_table_root_escaping(tmp_path: Path) -> None:
    import pickle
    import threading
    from urllib.parse import quote

    from deltalake import DeltaTable

    from tests.unit.io.cloud.conftest import CountingS3

    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    s3.client.create_bucket(Bucket="bucket")
    options = {
        "AWS_ACCESS_KEY_ID": s3.storage_options["aws_access_key_id"],
        "AWS_SECRET_ACCESS_KEY": s3.storage_options["aws_secret_access_key"],
        "AWS_REGION": s3.storage_options["aws_region"],
        "AWS_ENDPOINT_URL": s3.endpoint,
        "AWS_ALLOW_HTTP": "true",
    }
    expected = pl.DataFrame({"p": ["100%", "ä/ö", "a b"], "x": [1, 2, 3]})
    expected.write_delta(tmp_path, delta_write_options={"partition_by": "p"})
    try:
        for name in ["space root", "percent%20literal", "percent%literal", "ä root"]:
            # Upload exact raw keys: deltalake writers accept escaped URLs instead.
            for file in tmp_path.rglob("*"):
                if file.is_file():
                    s3.client.put_object(
                        Bucket="bucket",
                        Key=f"{name}/{file.relative_to(tmp_path).as_posix()}",
                        Body=file.read_bytes(),
                    )
            sources: list[Any] = [f"s3://bucket/{name}"]
            # Released deltalake double-decodes literal %20 roots during loading.
            # The raw native path still has to work for that exact object key.
            if name != "percent%20literal":
                sources.append(
                    DeltaTable(
                        f"s3://bucket/{quote(name, safe='')}", storage_options=options
                    )
                )
            for source in sources:
                scan = pl.scan_delta(source, storage_options=options)
                assert_frame_equal(scan.collect().sort("x"), expected)
                restored = pickle.loads(pickle.dumps(scan))
                assert_frame_equal(restored.collect().sort("x"), expected)
                assert scan.select(pl.len()).collect().item() == 3
                assert_frame_equal(
                    scan.filter(pl.col("p") == "100%").collect(), expected.head(1)
                )
    finally:
        s3.server.shutdown()


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("root", "twin"),
    [
        ("literal%20space", "literal space"),
        ("name#fragment", "name"),
        ("name?query", "name"),
    ],
)
def test_delta_plugin_cloud_root_fallback_preserves_identity(
    tmp_path: Path, root: str, twin: str
) -> None:
    import pickle
    import subprocess
    import sys
    import threading

    from tests.unit.io.cloud.conftest import CountingS3

    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    s3.client.create_bucket(Bucket="bucket")
    try:
        expected = pl.DataFrame({"a": [101]})
        other = pl.DataFrame({"a": [202]})
        for i, (name, data) in enumerate([(root, expected), (twin, other)]):
            local = _table(tmp_path / str(i), [_field("a", "long")], [data])
            for file in local.rglob("*"):
                if file.is_file():
                    s3.client.put_object(
                        Bucket="bucket",
                        Key=f"{name}/{file.relative_to(local).as_posix()}",
                        Body=file.read_bytes(),
                    )
        scan = pl.scan_delta(f"s3://bucket/{root}", storage_options=s3.storage_options)
        assert scan.collect_schema() == expected.schema
        serialized = pickle.dumps(scan)
        assert_frame_equal(scan.collect(), expected)
        assert_frame_equal(
            pl.scan_delta(
                f"s3://bucket/{twin}", storage_options=s3.storage_options
            ).collect(),
            other,
        )

        mark = len(s3.log)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
import pickle
import sys

import polars as pl

os.environ.pop("POLARS_DELTA_PLANNER", None)
sys.modules["polars_delta"] = None
scan = pickle.loads(sys.stdin.buffer.read())
try:
    result = scan.collect()
except pl.exceptions.ComputeError as exc:
    assert "deltalake cannot reliably reopen this native Delta cloud path" in str(exc)
else:
    raise AssertionError(f"read with an ambiguous cloud path: {result}")
""",
            ],
            input=serialized,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert s3.since(mark) == []
    finally:
        s3.server.shutdown()


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_deltalake_cloud_root_preserves_identity_with_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pickle
    import subprocess
    import sys
    import threading

    from deltalake import write_deltalake

    from tests.unit.io.cloud.conftest import CountingS3

    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    s3.client.create_bucket(Bucket="bucket")
    root, twin = "literal%20space", "literal space"
    options = {key.upper(): value for key, value in s3.storage_options.items()}
    expected = pl.DataFrame({"a": [202]})
    other = pl.DataFrame({"a": [101]})
    try:
        for i, (name, data) in enumerate([(root, other), (twin, expected)]):
            local = tmp_path / str(i)
            write_deltalake(local, data.to_arrow())
            for file in local.rglob("*"):
                if file.is_file():
                    key = file.relative_to(local).as_posix()
                    body = file.read_bytes()
                    s3.client.put_object(
                        Bucket="bucket", Key=f"{name}/{key}", Body=body
                    )
                    if name == twin and file.suffix == ".parquet":
                        # delta-rs uses the space-root log but returns escaped file
                        # URIs. Its uniquely named data file is readable either way.
                        s3.client.put_object(
                            Bucket="bucket", Key=f"{root}/{key}", Body=body
                        )

        uri = f"s3://bucket/{root}"
        # Explicit `plugin` opts into the raw native root: the literal key.
        assert_frame_equal(pl.scan_delta(uri, storage_options=options).collect(), other)
        # The default (auto) keeps the meaning released Polars gave the spelling:
        # the decoded prefix deltalake opens, the same as forcing deltalake.
        monkeypatch.delenv("POLARS_DELTA_PLANNER")
        assert_frame_equal(
            pl.scan_delta(uri, storage_options=options).collect(), expected
        )
        monkeypatch.setenv("POLARS_DELTA_PLANNER", "deltalake")
        scan = pl.scan_delta(uri, storage_options=options)
        assert scan.collect_schema() == expected.schema
        serialized = pickle.dumps(scan)
        assert_frame_equal(scan.collect(), expected)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
import pickle
import sys

from polars.io.delta._plugin import use_plugin_planner

os.environ.pop("POLARS_DELTA_PLANNER")
assert use_plugin_planner()
scan = pickle.loads(sys.stdin.buffer.read())
assert scan.collect().rows() == [(202,)]
""",
            ],
            input=serialized,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr.decode()
    finally:
        s3.server.shutdown()


@pytest.mark.parametrize(
    "encoded",
    [
        "2021-09-08 11:11:11.123456",
        "2021-09-08T11:11:11.123456Z",
        "2021-09-08T18:11:11.123456+07:00",
        "2021-09-08T04:11:11.123456-07:00",
    ],
)
def test_delta_plugin_timestamp_partition_offsets(tmp_path: Path, encoded: str) -> None:
    from datetime import UTC, datetime

    path = _table(
        tmp_path / "t",
        [_field("p", "timestamp"), _field("x", "long")],
        [pl.DataFrame({"x": [1]})],
        partition_columns=["p"],
        partitions=[{"p": encoded}],
    )
    timestamp = datetime(2021, 9, 8, 11, 11, 11, 123456, tzinfo=UTC)
    expected = pl.DataFrame({"p": [timestamp], "x": [1]})
    scan = pl.scan_delta(path)
    if encoded == "2021-09-08 11:11:11.123456":
        with pytest.raises(
            pl.exceptions.ComputeError, match="requires an explicit UTC offset"
        ):
            scan.collect()
        return
    assert_frame_equal(scan.collect(), expected)
    assert_frame_equal(scan.filter(pl.col("p") == timestamp).collect(), expected)
    assert scan.filter(pl.col("p") != timestamp).select(pl.len()).collect().item() == 0


def test_delta_plugin_decimal_statistics_are_exact(tmp_path: Path) -> None:
    from decimal import Decimal

    # Needs 19 significant digits: rounding through f64 would prune the file.
    big = Decimal("12345678901234567.89")
    frames = [pl.DataFrame({"d": [big]}), pl.DataFrame({"d": [Decimal("1.00")]})]
    frames = [f.cast({"d": pl.Decimal(20, 2)}) for f in frames]
    path = _table(tmp_path / "t", [_field("d", "decimal(20,2)")], frames)
    log = path / "_delta_log/00000000000000000000.json"
    actions = [json.loads(line) for line in log.read_text().splitlines()]
    for action, text in zip(
        (a for a in actions if "add" in a),
        ["12345678901234567.89", "1.00"],
        strict=True,
    ):
        action["add"]["stats"] = (
            f'{{"numRecords":1,"minValues":{{"d":{text}}},"maxValues":{{"d":{text}}},'
            '"nullCount":{"d":0}}'
        )
    log.write_text("\n".join(json.dumps(action) for action in actions))

    scan = pl.scan_delta(path)
    for predicate in [
        pl.col("d") == big,
        pl.col("d") >= big,
        pl.col("d") > big - 1,
        pl.col("d").is_between(big, big),
    ]:
        assert_frame_equal(scan.filter(predicate).collect(), frames[0])
    assert scan.filter(pl.col("d") > big).collect().height == 0


@pytest.mark.parametrize("unit", ["ms", "us", "ns", "int96"])
def test_delta_plugin_checkpoint_timestamp_statistics_unit(
    tmp_path: Path, unit: str
) -> None:
    from datetime import UTC, datetime

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    t = datetime(2024, 1, 1, 12, tzinfo=UTC)
    expected = pl.DataFrame({"t": [t]}).cast({"t": pl.Datetime("us", "UTC")})
    path = _table(tmp_path / "t", [_field("t", "timestamp")], [expected])
    log = path / "_delta_log"
    commit = log / "00000000000000000000.json"
    actions = [json.loads(line) for line in commit.read_text().splitlines()]
    commit.unlink()
    add = next(a["add"] for a in actions if "add" in a)
    metadata = next(a["metaData"] for a in actions if "metaData" in a)

    # Parquet checkpoints store timestamp bounds in the writer's unit (or INT96).
    ts = pa.timestamp("us" if unit == "int96" else unit, tz="UTC")
    bound = pa.struct([("t", ts)])
    map_type = pa.map_(pa.string(), pa.string())
    stats_type = pa.struct(
        [
            ("numRecords", pa.int64()),
            ("tightBounds", pa.bool_()),
            ("minValues", bound),
            ("maxValues", bound),
            ("nullCount", pa.struct([("t", pa.int64())])),
        ]
    )
    schema = pa.schema(
        [
            (
                "add",
                pa.struct(
                    [
                        ("path", pa.string()),
                        ("size", pa.int64()),
                        ("partitionValues", map_type),
                        ("stats_parsed", stats_type),
                    ]
                ),
            ),
            (
                "metaData",
                pa.struct(
                    [
                        ("id", pa.string()),
                        (
                            "format",
                            pa.struct(
                                [("provider", pa.string()), ("options", map_type)]
                            ),
                        ),
                        ("schemaString", pa.string()),
                        ("partitionColumns", pa.list_(pa.string())),
                        ("configuration", map_type),
                    ]
                ),
            ),
            (
                "protocol",
                pa.struct(
                    [
                        ("minReaderVersion", pa.int32()),
                        ("minWriterVersion", pa.int32()),
                    ]
                ),
            ),
        ]
    )
    rows = [
        {
            "add": {
                "path": add["path"],
                "size": add["size"],
                "partitionValues": [],
                "stats_parsed": {
                    "numRecords": 1,
                    "tightBounds": True,
                    "minValues": {"t": t},
                    "maxValues": {"t": t},
                    "nullCount": {"t": 0},
                },
            }
        },
        {
            "metaData": {
                "id": metadata["id"],
                "format": {"provider": "parquet", "options": []},
                "schemaString": metadata["schemaString"],
                "partitionColumns": [],
                "configuration": [],
            }
        },
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        log / "00000000000000000000.checkpoint.parquet",
        use_deprecated_int96_timestamps=unit == "int96",
    )

    scan = pl.scan_delta(path)
    for predicate in [
        pl.col("t") == t,
        pl.col("t") >= t,
        pl.col("t") > datetime(2024, 1, 1, tzinfo=UTC),
        pl.col("t") < datetime(2024, 1, 2, tzinfo=UTC),
    ]:
        assert_frame_equal(scan.filter(predicate).collect(), expected)


def test_delta_plugin_recreated_table_at_same_path(tmp_path: Path) -> None:
    import shutil

    path = tmp_path / "t"
    pl.DataFrame({"a": [1, 2]}).write_delta(path)
    held = pl.scan_delta(path)
    assert_frame_equal(held.collect(), pl.DataFrame({"a": [1, 2]}))
    # Same path and version (0), other files: nothing of the first table may be reused.
    shutil.rmtree(path)
    pl.DataFrame({"a": [7, 8, 9], "b": [1, 1, 1]}).write_delta(path)
    assert_frame_equal(
        pl.scan_delta(path).collect(), pl.DataFrame({"a": [7, 8, 9], "b": [1, 1, 1]})
    )
    # The held scan stays pinned to the first table's files: it fails, not mixes.
    with pytest.raises(FileNotFoundError):
        held.collect()


def test_delta_plugin_recreated_table_reuses_log_name_and_mtime(tmp_path: Path) -> None:
    import os
    import shutil

    import polars._plr as plr
    from polars._utils.wrap import wrap_ldf
    from polars.io.delta._plugin import _plugin_capsule

    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    commit = path / "_delta_log/00000000000000000000.json"
    old_commit = commit.read_text()
    st = commit.stat()
    cache = plr.PyIcebergMetadataFileCache(1 << 20)

    def collect() -> pl.DataFrame:
        ldf, _, _ = plr._delta_plugin_scan(
            _plugin_capsule(plr._IO_PLUGIN_IDS),
            table_uri=str(path),
            version=None,
            timestamp_ms=None,
            projection=None,
            filter_columns=None,
            statistics_columns=None,
            limit=None,
            use_metadata_statistics=True,
            verbose=False,
            testing_fail=None,
            storage_options=None,
            credential_provider=None,
            metadata_cache=cache,
        )
        return wrap_ldf(ldf).collect()

    assert_frame_equal(collect(), pl.DataFrame({"a": [1]}))
    shutil.rmtree(path)
    _table(path, [_field("a", "long")], [pl.DataFrame({"a": [2]})])
    (path / "part-0.parquet").rename(path / "part-1.parquet")
    commit.write_text(old_commit.replace("part-0.parquet", "part-1.parquet"))
    os.utime(commit, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert commit.stat().st_size == st.st_size
    assert commit.stat().st_mtime_ns == st.st_mtime_ns
    assert_frame_equal(collect(), pl.DataFrame({"a": [2]}))


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_plugin_checkpoint_replaced_between_head_and_range(
    tmp_path: Path,
) -> None:
    import threading
    from uuid import uuid4

    from deltalake import DeltaTable

    from tests.unit.io.cloud.conftest import CountingS3

    expected = pl.DataFrame({"a": range(20)})
    expected.write_delta(tmp_path)
    DeltaTable(tmp_path).create_checkpoint()
    pq = pytest.importorskip("pyarrow.parquet")
    checkpoint = tmp_path / "_delta_log/00000000000000000000.checkpoint.parquet"
    # Moto shares backend state across servers. Use a fresh bucket for this table.
    bucket = f"delta-checkpoint-race-{uuid4().hex}"
    key = checkpoint.relative_to(tmp_path).as_posix()
    request_path = f"{bucket}/{key}"
    # Equivalent actions with different physical layout and footer offset.
    replacement = tmp_path / "replacement.parquet"
    pq.write_table(pq.read_table(checkpoint), replacement, row_group_size=1)
    assert checkpoint.stat().st_size != replacement.stat().st_size
    replacement_bytes = replacement.read_bytes()
    replacement.unlink()

    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    replaced = threading.Event()

    def replace_before_range(environ: dict[str, Any]) -> None:
        if (
            environ["REQUEST_METHOD"] == "GET"
            and environ.get("PATH_INFO") == f"/{request_path}"
            and environ.get("HTTP_RANGE")
            and not replaced.is_set()
        ):
            s3.client.put_object(Bucket=bucket, Key=key, Body=replacement_bytes)
            replaced.set()

    try:
        s3.client.create_bucket(Bucket=bucket)
        for file in tmp_path.rglob("*"):
            if file.is_file():
                s3.client.put_object(
                    Bucket=bucket,
                    Key=file.relative_to(tmp_path).as_posix(),
                    Body=file.read_bytes(),
                )
        s3.on_request = replace_before_range
        scan = pl.scan_delta(f"s3://{bucket}/", storage_options=s3.storage_options)
        assert_frame_equal(scan.collect(), expected)
        assert replaced.is_set()
        ranges = [
            r for method, path, r in s3.log if method == "GET" and path == request_path
        ]
        # The checkpoint was read with ranges before and after replacement.
        assert len(ranges) > 1
        assert all(ranges)
    finally:
        s3.server.shutdown()


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_plugin_lakefs_reads_through_s3_gateway(tmp_path: Path) -> None:
    import threading

    from tests.unit.io.cloud.conftest import CountingS3

    first = pl.DataFrame({"a": [1, 2]})
    second = pl.DataFrame({"a": [3]})
    first.write_delta(tmp_path)
    second.write_delta(tmp_path, mode="append")

    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    try:
        s3.client.create_bucket(Bucket="repo")
        for file in tmp_path.rglob("*"):
            if file.is_file():
                s3.client.put_object(
                    Bucket="repo",
                    Key=f"main/t/{file.relative_to(tmp_path).as_posix()}",
                    Body=file.read_bytes(),
                )
        options = s3.storage_options
        for version, expected in [(None, pl.concat([first, second])), (0, first)]:
            lakefs = pl.scan_delta(
                "lakefs://repo/main/t", version=version, storage_options=options
            )
            s3_scan = pl.scan_delta(
                "s3://repo/main/t", version=version, storage_options=options
            )
            assert_frame_equal(lakefs.collect(), s3_scan.collect())
            assert_frame_equal(lakefs.collect(), expected)
    finally:
        s3.server.shutdown()


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_plugin_missing_last_checkpoint_does_not_rebuild_credentials(
    tmp_path: Path,
) -> None:
    import threading

    from tests.unit.io.cloud.conftest import CountingS3

    pl.DataFrame({"a": [1]}).write_delta(tmp_path)
    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    calls = 0
    creds = {
        "aws_access_key_id": s3.storage_options["aws_access_key_id"],
        "aws_secret_access_key": s3.storage_options["aws_secret_access_key"],
    }

    def provider() -> tuple[dict[str, str], int | None]:
        nonlocal calls
        calls += 1
        return creds, None

    try:
        s3.client.create_bucket(Bucket="credbucket")
        for file in tmp_path.rglob("*"):
            if file.is_file():
                s3.client.put_object(
                    Bucket="credbucket",
                    Key=f"t/{file.relative_to(tmp_path).as_posix()}",
                    Body=file.read_bytes(),
                )
        options = {k: v for k, v in s3.storage_options.items() if "access" not in k}
        scan = pl.scan_delta(
            "s3://credbucket/t",
            storage_options=options,
            credential_provider=provider,
        )
        mark = len(s3.log)
        scan.collect_schema()
        last = [
            e
            for e in s3.log[mark:]
            if e[0] == "GET" and e[1] == "credbucket/t/_delta_log/_last_checkpoint"
        ]
        assert len(last) == 1
        assert calls == 1
    finally:
        s3.server.shutdown()


def _put_table(s3: Any, bucket: str, folder: Path) -> None:
    for file in folder.rglob("*"):
        if file.is_file():
            s3.client.put_object(
                Bucket=bucket,
                Key=file.relative_to(folder).as_posix(),
                Body=file.read_bytes(),
            )


def _write_versions(folder: Path, checkpoint: bool) -> None:
    from deltalake import DeltaTable

    pl.DataFrame({"a": [1, 2]}).write_delta(folder)
    pl.DataFrame({"a": [3]}).write_delta(folder, mode="append")
    if checkpoint:
        DeltaTable(folder).create_checkpoint()


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.parametrize("checkpoint", [True, False])
def test_delta_plugin_schema_then_collect_reads_hint_once(
    tmp_path: Path, checkpoint: bool
) -> None:
    import threading
    from uuid import uuid4

    from tests.unit.io.cloud.conftest import CountingS3

    _write_versions(tmp_path, checkpoint)
    bucket = f"delta-hint-once-{uuid4().hex}"
    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    try:
        s3.client.create_bucket(Bucket=bucket)
        _put_table(s3, bucket, tmp_path)
        scan = pl.scan_delta(f"s3://{bucket}/", storage_options=s3.storage_options)
        mark = len(s3.log)
        scan.collect_schema()
        assert scan.collect().sort("a")["a"].to_list() == [1, 2, 3]
        log = s3.since(mark)
        hint = f"{bucket}/_delta_log/_last_checkpoint"
        # Both plans, schema-only (unpinned) and full (pinned), read the hint once.
        assert [e for e in log if e[:2] == ("GET", hint)] == [("GET", hint, "")]
        # A checkpoint's size and generation come from the LIST: no HEAD.
        assert not [e for e in log if e[0] == "HEAD" and "checkpoint.parquet" in e[1]]
        # A new scan is unpinned and reads the hint again.
        mark = len(s3.log)
        fresh = pl.scan_delta(f"s3://{bucket}/", storage_options=s3.storage_options)
        assert fresh.collect().height == 3
        assert len([e for e in s3.since(mark) if e[:2] == ("GET", hint)]) == 1
    finally:
        s3.server.shutdown()


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_plugin_replaced_table_with_cached_hint(tmp_path: Path) -> None:
    import threading
    from uuid import uuid4

    from tests.unit.io.cloud.conftest import CountingS3

    old, new = tmp_path / "old", tmp_path / "new"
    _write_versions(old, checkpoint=True)
    _write_versions(new, checkpoint=False)
    bucket = f"delta-hint-replaced-{uuid4().hex}"
    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    try:
        s3.client.create_bucket(Bucket=bucket)
        _put_table(s3, bucket, old)
        scan = pl.scan_delta(f"s3://{bucket}/", storage_options=s3.storage_options)
        scan.collect_schema()  # caches the old table's hint, a checkpoint at version 1
        for obj in s3.client.list_objects_v2(Bucket=bucket)["Contents"]:
            s3.client.delete_object(Bucket=bucket, Key=obj["Key"])
        # The replacement has the same version but no checkpoint at the hint's version.
        _put_table(s3, bucket, new)
        mark = len(s3.log)
        with pytest.raises(pl.exceptions.ComputeError, match="replaced"):
            scan.collect()
        log = s3.since(mark)
        # The cached hint was served and failed: Kernel's plan listed the log, then the
        # retry read `_last_checkpoint` fresh (a 404) and listed again.
        hint = f"{bucket}/_delta_log/_last_checkpoint"
        assert len([e for e in log if e[:2] == ("GET", hint)]) == 1
        assert len([e for e in log if e[0] == "GET" and e[1] == bucket]) == 2
    finally:
        s3.server.shutdown()


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_plugin_advancing_supplied_table_reads_hint_fresh(
    tmp_path: Path,
) -> None:
    import threading
    from uuid import uuid4

    from deltalake import DeltaTable

    from tests.unit.io.cloud.conftest import CountingS3

    _write_versions(tmp_path, checkpoint=True)
    bucket = f"delta-hint-advance-{uuid4().hex}"
    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    try:
        s3.client.create_bucket(Bucket=bucket)
        _put_table(s3, bucket, tmp_path)
        uri = f"s3://{bucket}/"
        table = DeltaTable(uri, storage_options=s3.storage_options)
        scan = pl.scan_delta(table, storage_options=s3.storage_options)
        scan.collect_schema()
        pl.DataFrame({"a": [4]}).write_delta(tmp_path, mode="append")
        _put_table(s3, bucket, tmp_path)
        table.update_incremental()
        assert table.version() == 2
        mark = len(s3.log)
        assert scan.collect().height == 4
        hint = f"{bucket}/_delta_log/_last_checkpoint"
        # The plan for version 2 does not reuse the hint recorded for version 1.
        assert len([e for e in s3.since(mark) if e[:2] == ("GET", hint)]) == 1
        # Another plan for version 2 (another filter) does.
        mark = len(s3.log)
        assert scan.filter(pl.col("a") > 1).collect().height == 3
        log = s3.since(mark)
        assert [e for e in log if e[0] == "GET" and e[1] == bucket]  # it did plan
        assert not [e for e in log if e[:2] == ("GET", hint)]
    finally:
        s3.server.shutdown()


def _add_parquet(path: Path, table: Any) -> None:
    import pyarrow.parquet as pq

    file = path / "data.parquet"
    pq.write_table(table, file)
    with (path / "_delta_log/00000000000000000000.json").open("a") as f:
        f.write(
            "\n"
            + json.dumps(
                {
                    "add": {
                        "path": file.name,
                        "size": file.stat().st_size,
                        "partitionValues": {},
                        "dataChange": True,
                    }
                }
            )
        )


@pytest.mark.parametrize(
    ("physical", "expected_key"),
    [
        (("a", "b"), {"a": 10, "b": 20}),
        (("b", "a"), {"a": 20, "b": 10}),
        (("phys_a", "phys_b"), {"a": 10, "b": 20}),
    ],
)
def test_delta_plugin_mapped_map_struct_keys(
    tmp_path: Path, physical: tuple[str, str], expected_key: dict[str, int]
) -> None:
    import pyarrow as pa

    def field_id(i: int) -> dict[bytes, bytes]:
        return {b"PARQUET:field_id": str(i).encode()}

    # The file holds the physical fields in the order of `physical_names`.
    physical_names = ["a", "b"] if set(physical) == {"a", "b"} else list(physical)
    ids = {physical[0]: 2, physical[1]: 3}
    key = pa.struct(
        [pa.field(n, pa.int64(), metadata=field_id(ids[n])) for n in physical_names]
    )
    dtype = pa.map_(key, pa.int64())
    schema = pa.schema([pa.field("phys_map", dtype, metadata=field_id(1))])
    values = dict(zip(physical_names, (10, 20), strict=True))
    data = pa.Table.from_arrays([pa.array([[(values, 5)]], type=dtype)], schema=schema)

    def mapped(name: str, dtype: Any, phys: str, i: int) -> dict[str, Any]:
        return _field(
            name,
            dtype,
            **{"delta.columnMapping.id": i, "delta.columnMapping.physicalName": phys},
        )

    delta_key = {
        "type": "struct",
        "fields": [
            mapped("a", "long", physical[0], 2),
            mapped("b", "long", physical[1], 3),
        ],
    }
    delta_map = {
        "type": "map",
        "keyType": delta_key,
        "valueType": "long",
        "valueContainsNull": True,
    }
    path = _table(
        tmp_path / "t", [mapped("m", delta_map, "phys_map", 1)], [], mode="name"
    )
    _add_parquet(path, data)

    lf = pl.scan_delta(path)
    for engine in ("auto", "streaming"):
        out = lf.collect(engine=engine)
        assert out.to_arrow()["m"].to_pylist() == [[(expected_key, 5)]]


@pytest.mark.parametrize("mode", ["id", "name"])
def test_delta_plugin_map_declared_nested_ids(tmp_path: Path, mode: str) -> None:
    import pyarrow as pa

    key = pa.field(
        "key", pa.string(), nullable=False, metadata={b"PARQUET:field_id": b"102"}
    )
    value = pa.field("value", pa.int64(), metadata={b"PARQUET:field_id": b"103"})
    dtype = pa.map_(key, value)
    schema = pa.schema(
        [pa.field("phys_map", dtype, metadata={b"PARQUET:field_id": b"2"})]
    )
    maps = [[("a", 7), ("b", None)], [], None]
    data = pa.Table.from_arrays([pa.array(maps, type=dtype)], schema=schema)

    delta_map = {
        "type": "map",
        "keyType": "string",
        "valueType": "long",
        "valueContainsNull": True,
    }
    field = _field(
        "m",
        delta_map,
        **{
            "delta.columnMapping.id": 2,
            "delta.columnMapping.physicalName": "phys_map",
            "delta.columnMapping.nested.ids": {
                "phys_map.key": 102,
                "phys_map.value": 103,
            },
        },
    )
    path = _table(tmp_path / "t", [field], [], mode=mode)
    _add_parquet(path, data)

    out = pl.scan_delta(path).collect()
    assert out.to_arrow()["m"].to_pylist() == maps


@pytest.mark.write_disk
@pytest.mark.parametrize("held", ["schema", "projection", "supplied"])
def test_delta_plugin_refuses_replaced_table_at_same_version(
    tmp_path: Path, held: str
) -> None:
    pa = pytest.importorskip("pyarrow")
    deltalake = pytest.importorskip("deltalake")

    path, old = tmp_path / "table", tmp_path / "old"
    deltalake.write_deltalake(path, pa.table({"x": [1], "y": [2]}))
    if held == "supplied":
        scan = pl.scan_delta(deltalake.DeltaTable(path))
    else:
        scan = pl.scan_delta(path)
        if held == "schema":
            scan.collect_schema()
        else:
            assert scan.collect().rows() == [(1, 2)]
    path.rename(old)
    deltalake.write_deltalake(path, pa.table({"x": [9], "y": [8]}))
    assert deltalake.DeltaTable(path).version() == 0
    # A fresh scan reads the replacement; the held one must not.
    assert pl.scan_delta(path).collect().rows() == [(9, 8)]
    with pytest.raises(pl.exceptions.ComputeError, match="replaced"):
        (scan.select("x") if held == "projection" else scan).collect()


@pytest.mark.parametrize(
    "options",
    [
        {"without_files": True},
        {"log_buffer_size": 4},
        {"skip_stats": True},
        {"without_files": True, "log_buffer_size": 4, "skip_stats": True},
    ],
)
def test_delta_plugin_ignores_noop_delta_table_options(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    expected = pl.scan_delta(path).collect()
    assert_frame_equal(
        pl.scan_delta(path, delta_table_options=options).collect(), expected
    )


def test_delta_plugin_unknown_delta_table_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    options = {"without_files": True, "bogus": 1}
    with pytest.raises(NotImplementedError, match="bogus"):
        pl.scan_delta(path, delta_table_options=options).collect()
    monkeypatch.delenv("POLARS_DELTA_PLANNER")
    with pytest.raises(TypeError):
        pl.scan_delta(path, delta_table_options=options).collect()


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_plugin_lakefs_absolute_add_path(tmp_path: Path) -> None:
    import json
    import threading

    from tests.unit.io.cloud.conftest import CountingS3

    expected = pl.DataFrame({"a": [1, 2]})
    expected.write_delta(tmp_path)

    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    try:
        s3.client.create_bucket(Bucket="lakefs-abs")
        for file in tmp_path.rglob("*"):
            if not file.is_file():
                continue
            relative = file.relative_to(tmp_path).as_posix()
            body = file.read_bytes()
            if relative.endswith(".json"):
                actions = [json.loads(line) for line in body.decode().splitlines()]
                for action in actions:
                    if "add" in action:
                        path = action["add"]["path"]
                        action["add"]["path"] = f"lakefs://lakefs-abs/main/abs/{path}"
                body = "\n".join(json.dumps(a) for a in actions).encode()
            s3.client.put_object(
                Bucket="lakefs-abs", Key=f"main/abs/{relative}", Body=body
            )
        scan = pl.scan_delta(
            "lakefs://lakefs-abs/main/abs", storage_options=s3.storage_options
        )
        assert_frame_equal(scan.collect(), expected)
    finally:
        s3.server.shutdown()
