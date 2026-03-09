from __future__ import annotations

import json
import struct
import uuid
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pyroaring import BitMap

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch

# NOTE
# This file contains temporary homegrown logic with the sole purpose of generating
# deletion vectors to automate delta reader capability testing on CI. It is
# explicitly not comprehensive and should be used with care.
# Any test case should compare the result with the outcome of a supported reader.
# In doubt, an alternate writer should be considered (e.g., pyspark), and the
# protocol spec should be taken into account.
#
# The intent is to replace this writer with a delta-rs supported implementation
# when available.

# Ref
# https://github.com/delta-io/delta/blob/master/PROTOCOL.md#deletion-vector-format
# See also delta-kernel deserialize

#
# Encode & serialize
#


def z85_encode(data: bytes) -> str:
    alphabet = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
    assert len(data) % 4 == 0
    result = bytearray()
    for i in range(0, len(data), 4):
        value = struct.unpack(">I", data[i : i + 4])[0]
        chunk = bytearray(5)
        for j in range(4, -1, -1):
            chunk[j] = alphabet[value % 85]
            value //= 85
        result.extend(chunk)
    return result.decode("ascii")


def serialize_roaring_bitmap_array(deleted_rows: list[int]) -> bytes:
    # format:
    #     magic(4LE) + numBuckets(8LE) + [key(4LE) + bucketData]*
    MAGIC_NUMBER = 1681511377

    groups: dict[int, BitMap] = {}
    for row in deleted_rows:
        high = row >> 32
        low = row & 0xFFFFFFFF
        if high not in groups:
            groups[high] = BitMap()
        groups[high].add(low)

    out = bytearray()
    out += struct.pack("<I", MAGIC_NUMBER)
    out += struct.pack("<Q", len(groups))
    for high, bm in sorted(groups.items()):
        out += struct.pack("<I", high)
        out += bm.serialize()

    return bytes(out)


def serialize_dv(deleted_rows: list[int]) -> tuple[bytes, int]:
    """Returns (file_bytes, bitmap_data_size)."""
    # Format:
    #   dataSize(4BE) + bitmapData(nLE) + checksum(4BE)
    bitmap_data = serialize_roaring_bitmap_array(deleted_rows)
    data_size = len(bitmap_data)
    checksum = zlib.crc32(bitmap_data) & 0xFFFFFFFF
    out = bytearray()
    out += struct.pack(">I", data_size)
    out += bitmap_data
    out += struct.pack(">I", checksum)
    return bytes(out), data_size


def uuid_to_filename(dv_uuid: uuid.UUID) -> str:
    return f"deletion_vector_{dv_uuid}.bin"


def uuid_to_z85(dv_uuid: uuid.UUID) -> str:
    """Encode UUID bytes as Z85 (16 bytes -> 20 chars)."""
    return z85_encode(dv_uuid.bytes)


def pa_schema_to_delta_schema(schema: pa.Schema) -> str:
    """Convert PyArrow schema to Delta Lake schema JSON string."""

    def pa_type_to_delta(t: pa.DataType) -> str:
        if pa.types.is_int32(t):
            return "integer"
        if pa.types.is_int64(t):
            return "long"
        if pa.types.is_float32(t):
            return "float"
        if pa.types.is_float64(t):
            return "double"
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return "string"
        if pa.types.is_boolean(t):
            return "boolean"
        msg = f"Unsupported type: {t}"
        raise ValueError(msg)

    fields = [
        {
            "name": field.name,
            "type": pa_type_to_delta(field.type),
            "nullable": True,
            "metadata": {},
        }
        for field in schema
    ]
    return json.dumps(
        {
            "type": "struct",
            "fields": fields,
        }
    )


#
# Statistics
#


def _arrow_scalar_to_json(scalar: pa.Scalar) -> object:
    if scalar is None or not scalar.is_valid:
        return None
    v = scalar.as_py()
    # PyArrow may return numpy scalars; cast to plain Python.
    if hasattr(v, "item"):
        return v.item()
    return v


class DeltaStats(TypedDict):
    """Delta table statistics."""

    numRecords: int
    tightBounds: bool
    minValues: dict[str, object]
    maxValues: dict[str, object]
    nullCount: dict[str, int]


def compute_stats(table: pa.Table, tight_bounds: bool = True) -> DeltaStats:
    """
    Compute Delta-compatible statistics for *table*.

    Returns a dict suitable for ``json.dumps`` insertion into the ``stats``
    field of an ``add`` action::

        {
            "numRecords": <int>,
            "minValues":  {col: value, ...},
            "maxValues":  {col: value, ...},
            "nullCount":  {col: int,   ...},
        }

    Columns whose type does not support min/max (e.g. struct, list) are
    silently skipped in minValues / maxValues but still appear in nullCount.

    Note: statistics describe the *physical* file — they are NOT filtered by
    any deletion vector.  Delta readers are expected to account for this.
    """
    num_records = len(table)
    min_values: dict[str, object] = {}
    max_values: dict[str, object] = {}
    null_count: dict[str, int] = {}

    for name in table.schema.names:
        col = table.column(name)
        field = table.schema.field(name)
        t = field.type

        # null count is always cheap
        null_count[name] = pc.sum(pc.is_null(col)).as_py() or 0

        # min / max only for primitive comparable types
        if (
            pa.types.is_integer(t)
            or pa.types.is_floating(t)
            or pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_date(t)
            or pa.types.is_timestamp(t)
        ):
            try:
                mn = _arrow_scalar_to_json(pc.min(col))
                mx = _arrow_scalar_to_json(pc.max(col))
                if mn is not None:
                    min_values[name] = mn
                if mx is not None:
                    max_values[name] = mx
            except Exception:
                # unsupported column type for min/max: skip silently
                pass

    return {
        "numRecords": num_records,
        "tightBounds": tight_bounds,
        "minValues": min_values,
        "maxValues": max_values,
        "nullCount": null_count,
    }


#
# Table creation
#


def create_dv_table(
    table_path: str | Path,
    data: pa.Table,
    deleted_rows: list[int],
) -> None:
    """
    Create a Delta table with a deletion vector.

    Args:
        table_path: Path to create the Delta table at.
        data: PyArrow table with the initial data.
        deleted_rows: Row indices to mark as deleted.
    """
    table_path = Path(table_path)
    table_path.mkdir(parents=True)

    delta_log = table_path / "_delta_log"
    delta_log.mkdir()

    # Write parquet file
    parquet_filename = "part-00000-test.snappy.parquet"
    parquet_path = table_path / parquet_filename
    pq.write_table(data, parquet_path, compression="snappy")

    parquet_size = parquet_path.stat().st_size
    stats = compute_stats(data, tight_bounds=False)

    # --- Commit 0: initial table setup ---
    commit_0 = {
        "protocol": {
            "minReaderVersion": 3,
            "minWriterVersion": 7,
            "readerFeatures": ["deletionVectors"],
            "writerFeatures": ["deletionVectors"],
        }
    }
    commit_0_metadata = {
        "metaData": {
            "id": str(uuid.uuid4()),
            "format": {"provider": "parquet", "options": {}},
            "schemaString": pa_schema_to_delta_schema(data.schema),
            "partitionColumns": [],
            "configuration": {"delta.enableDeletionVectors": "true"},
            "createdTime": 0,
        }
    }
    commit_0_add = {
        "add": {
            "path": parquet_filename,
            "partitionValues": {},
            "size": parquet_size,
            "modificationTime": 0,
            "dataChange": True,
            "stats": json.dumps(stats),
        }
    }

    with Path.open(delta_log / "00000000000000000000.json", "w") as f:
        f.write(json.dumps(commit_0) + "\n")
        f.write(json.dumps(commit_0_metadata) + "\n")
        f.write(json.dumps(commit_0_add) + "\n")

    # --- Commit 1: add deletion vector ---
    dv_uuid = uuid.uuid4()
    dv_filename = uuid_to_filename(dv_uuid)
    dv_path = table_path / dv_filename

    dv_bytes, bitmap_data_size = serialize_dv(deleted_rows)
    with Path.open(dv_path, "wb") as f:
        f.write(b"\x01")  # version byte
        f.write(dv_bytes)

    commit_1_remove = {
        "remove": {
            "path": parquet_filename,
            "partitionValues": {},
            "deletionTimestamp": 1000,
            "dataChange": True,
        }
    }
    commit_1_add = {
        "add": {
            "path": parquet_filename,
            "partitionValues": {},
            "size": parquet_size,
            "modificationTime": 0,
            "dataChange": True,
            # Stats on the re-added action still describe the physical file
            # (pre-deletion).  Delta readers combine stats + DV for pruning.
            "stats": json.dumps(stats),
            "deletionVector": {
                "storageType": "u",
                "pathOrInlineDv": uuid_to_z85(dv_uuid),
                "offset": 1,
                "sizeInBytes": bitmap_data_size,
                "cardinality": len(deleted_rows),
            },
        }
    }

    with Path.open(delta_log / "00000000000000000001.json", "w") as f:
        f.write(json.dumps(commit_1_remove) + "\n")
        f.write(json.dumps(commit_1_add) + "\n")


def create_dv_table_multi(
    table_path: str | Path,
    files: list[tuple[pa.Table, list[int]]],  # (data, deleted_rows) per file
) -> None:
    table_path = Path(table_path)
    table_path.mkdir(parents=True)
    delta_log = table_path / "_delta_log"
    delta_log.mkdir()

    commit_0_actions = [
        {
            "protocol": {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": ["deletionVectors"],
                "writerFeatures": ["deletionVectors"],
            }
        },
        {
            "metaData": {
                "id": str(uuid.uuid4()),
                "format": {"provider": "parquet", "options": {}},
                "schemaString": pa_schema_to_delta_schema(files[0][0].schema),
                "partitionColumns": [],
                "configuration": {"delta.enableDeletionVectors": "true"},
                "createdTime": 0,
            }
        },
    ]

    commit_1_actions = []

    for i, (data, deleted_rows) in enumerate(files):
        parquet_filename = f"part-{i:05d}-test.snappy.parquet"
        parquet_path = table_path / parquet_filename
        pq.write_table(data, parquet_path, compression="snappy")

        stats = compute_stats(data, tight_bounds=False)

        commit_0_actions.append(
            {
                "add": {
                    "path": parquet_filename,
                    "partitionValues": {},
                    "size": parquet_path.stat().st_size,
                    "modificationTime": 0,
                    "dataChange": True,
                    "stats": json.dumps(stats),
                }
            }
        )

        if not deleted_rows:
            continue

        dv_uuid = uuid.uuid4()
        dv_bytes, bitmap_data_size = serialize_dv(deleted_rows)
        with Path.open(table_path / uuid_to_filename(dv_uuid), "wb") as f:
            f.write(b"\x01")
            f.write(dv_bytes)

        commit_1_actions.append(
            {
                "remove": {
                    "path": parquet_filename,
                    "partitionValues": {},
                    "deletionTimestamp": 1000,
                    "dataChange": True,
                }
            }
        )
        commit_1_actions.append(
            {
                "add": {
                    "path": parquet_filename,
                    "partitionValues": {},
                    "size": parquet_path.stat().st_size,
                    "modificationTime": 0,
                    "dataChange": True,
                    "stats": json.dumps(stats),
                    "deletionVector": {
                        "storageType": "u",
                        "pathOrInlineDv": uuid_to_z85(dv_uuid),
                        "offset": 1,
                        "sizeInBytes": bitmap_data_size,  # kdn
                        "cardinality": len(deleted_rows),
                    },
                }
            }
        )

    with Path.open(delta_log / "00000000000000000000.json", "w") as f:
        for action in commit_0_actions:
            f.write(json.dumps(action) + "\n")

    if commit_1_actions:
        with Path.open(delta_log / "00000000000000000001.json", "w") as f:
            for action in commit_1_actions:
                f.write(json.dumps(action) + "\n")


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("n_rows", "dv"),
    [
        (1, []),
        (1, [0]),
        (5, [2]),
        (5, [0]),
        (5, [4]),
        (10, [1, 3, 7]),
        (10, []),
        (10, list(range(10))),
    ],
)
def test_scan_delta_dv_single(
    n_rows: int,
    dv: list[int],
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_DELTA_READER_FEATURE_DV", "1")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    path = tmp_path / "delta_table"
    df = pl.DataFrame({"a": range(n_rows), "b": [f"b_{i}" for i in range(n_rows)]})
    data = df.to_arrow()
    create_dv_table(path, data, dv)

    out = pl.scan_delta(path).collect()
    capture = capfd.readouterr().err

    expected = df.with_row_index().filter(~pl.col.index.is_in(dv)).drop("index")

    assert_frame_equal(out, expected)

    # kdn TODO: known issue: no message is printed when the resulting df is empty
    if n_rows - len(dv) > 0:
        expected_msg = (
            f"DeltaDeletionVector(<{len(dv)} deletion{'' if len(dv) == 1 else 's'}>)"
        )
        assert expected_msg in capture

        rows_before = df.height
        rows_after = expected.height
        expected_msg = (
            f"[PostApplyExtraOps]: rows_before: {rows_before}, rows_after: {rows_after}"
        )
        assert expected_msg in capture


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("n_files", "n_rows", "dvs"),
    [
        (3, 5, [[3], [], [1, 4]]),
        (3, 3, [[], [], []]),
        (3, 3, [[0, 1, 2], [], []]),
        (3, 3, [[], [0, 1, 2], []]),
        (3, 3, [[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
    ],
)
def test_scan_delta_dv_multiple(
    n_files: int,
    n_rows: int,
    dvs: list[list[int]],
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_DELTA_READER_FEATURE_DV", "1")

    dfs = []
    for i in range(n_files):
        start = i * n_rows
        df = pl.DataFrame({"a": range(start, start + n_rows)})
        dfs.append(df)

    data = [df.to_arrow() for df in dfs]

    path = tmp_path / "delta_table"
    create_dv_table_multi(path, list(zip(data, dvs, strict=True)))

    out = pl.scan_delta(path).collect()

    expected = pl.concat(
        [
            df.with_row_index().filter(~pl.col.index.is_in(dv)).drop("index")
            for df, dv in zip(dfs, dvs, strict=True)
        ]
    )

    assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("n_files", "n_rows", "dvs"),
    [
        (3, 5, [[1, 2, 3], [], [2]]),
        (3, 5, [[], [0, 1, 2], [2]]),
        (3, 5, [[], [2], [1, 2, 3]]),
        (3, 5, [[], [], []]),
        (3, 5, [list(range(5)), list(range(5)), list(range(5))]),
    ],
)
def test_scan_delta_dv_multiple_with_predicate_pushdown(
    n_files: int,
    n_rows: int,
    dvs: list[list[int]],
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_DELTA_READER_FEATURE_DV", "1")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    dfs = []
    for i in range(n_files):
        start = i * n_rows
        df = pl.DataFrame({"a": range(start, start + n_rows)})
        df = df.with_columns((pl.col.a * 10).alias("b"))
        dfs.append(df)

    data = [df.to_arrow() for df in dfs]

    path = tmp_path / "delta_table"
    create_dv_table_multi(path, list(zip(data, dvs, strict=True)))

    for limit in range(0, n_files * n_rows * 10, 10):
        expr = pl.col.b >= limit
        out = pl.scan_delta(path).filter(expr).collect()
        capture = capfd.readouterr().err

        n_skip_files = sum(df.select((~expr).all()).item() for df in dfs)
        expected_msg = f"skipping {n_skip_files} / 3 files"

        assert expected_msg in capture

        expected = pl.concat(
            [
                df.with_row_index()
                .filter(~pl.col.index.is_in(dv))
                .drop("index")
                .filter(expr)
                for df, dv in zip(dfs, dvs, strict=True)
            ]
        )

        assert_frame_equal(out, expected, check_row_order=False)
