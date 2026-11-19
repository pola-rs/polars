from __future__ import annotations

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
    from polars.io.delta import _plugin

    path = _table(tmp_path / "t", [_field("a", "long")], [pl.DataFrame({"a": [1]})])
    if not required:
        monkeypatch.delenv("POLARS_DELTA_PLANNER")
    original_import = _plugin.importlib.import_module

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "polars_delta":
            msg = "No module named 'polars_delta'"
            raise ModuleNotFoundError(msg, name="polars_delta")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(_plugin.importlib, "import_module", missing)
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
        delta_type = {
            "type": "struct",
            "fields": [
                _field("inner", "decimal(4,1)", **{"delta.typeChanges": [change]})
            ],
        }
        dtype = pa.struct([pa.field("inner", pa.int8())])
        values = [{"inner": 12}]
        target = pl.Struct({"inner": pl.Decimal(4, 1)})
        metadata = {}
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
    elif change == "limit":
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

    version: str | datetime = datetime(2100, 1, 1, tzinfo=UTC)
    if selector == "string":
        version = version.isoformat()
    elif selector == "naive_datetime":
        version = version.replace(tzinfo=None)
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


@pytest.mark.slow
@pytest.mark.write_disk
def test_delta_plugin_cloud_table_root_escaping(tmp_path: Path) -> None:
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
                assert scan.select(pl.len()).collect().item() == 3
                assert_frame_equal(
                    scan.filter(pl.col("p") == "100%").collect(), expected.head(1)
                )
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
