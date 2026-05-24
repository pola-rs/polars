from __future__ import annotations

import functools
import json
import struct
import sys
import uuid
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from deltalake import DeltaTable
from pyroaring import BitMap  # type: ignore[import-not-found]

import polars as pl
from polars.io.delta._dataset import _extract_delta_deletion_vectors
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

        # null_count
        null_count[name] = col.null_count

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

    # Set statistics to pre-deletion-vector values
    # TODO: expand logic and test case to cover tight_bounds=True; this
    # requires statistics to be recomputed after applying the deletion vectors
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
                        "sizeInBytes": bitmap_data_size,
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


#
# Test suite: internal py methods
#


@pytest.mark.parametrize(
    ("requested_paths", "dvs", "expected_vectors"),
    [
        (["a", "b"], {"b": [False], "a": [True]}, [[True], [False]]),
        (["a", "c"], {"a": [False], "b": [False]}, [[False], None]),
        (["c", "d"], {"a": [False], "b": [False]}, [None, None]),
        ([], {"a": [False]}, []),
        (["a", "b"], {}, [None, None]),
        (["b"], {"a": [True], "b": [False]}, [[False]]),
        (["a", "a"], {"a": [False]}, [[False], [False]]),  # duplicate
    ],
)
def test_scan_delta_dv_extract_dvs(
    requested_paths: list[str],
    dvs: dict[str, list[bool]],
    expected_vectors: list[list[bool] | None],
) -> None:
    requested_df = pl.DataFrame({"path": requested_paths}, schema={"path": pl.String})
    delta_deletion_vectors = pl.DataFrame(
        {
            "filepath": list(dvs.keys()),
            "selection_vector": list(dvs.values()),
        },
        schema={"filepath": pl.String, "selection_vector": pl.List(pl.Boolean)},
    )
    out = _extract_delta_deletion_vectors(requested_df, delta_deletion_vectors)
    expected = pl.DataFrame(
        {"selection_vector": expected_vectors},
        schema_overrides={"selection_vector": pl.List(pl.Boolean)},
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("platform", "requested_paths", "dv_paths", "n_matches"),
    [
        # common
        (None, ["s3:///tmp/foo"], ["s3:///tmp/foo"], 1),
        (None, ["s3:///tmp/foo"], ["lakefs:///tmp/foo"], 1),
        (None, ["lakefs:///tmp/foo"], ["s3:///tmp/foo"], 1),
        (None, ["lakefs:///tmp/foo"], ["lakefs:///tmp/foo"], 1),
        # posix
        ("posix", ["/tmp/foo"], ["file:///tmp/foo"], 1),
        ("posix", ["file:///tmp/foo"], ["file:///tmp/foo"], 1),
        ("posix", ["/tmp/foo"], ["/tmp/foo"], 1),
        ("posix", ["/tmp/foo"], ["s3:///tmp/foo"], 0),
        ("posix", ["file:///tmp/foo"], ["s3:///tmp/foo"], 0),
        # win32
        ("win32", ["C:/foo"], ["file:///C:/foo"], 1),
        ("win32", ["file:///C:/foo"], ["file:///C:/foo"], 1),
        ("win32", ["C:/foo"], ["C:/foo"], 1),
        ("win32", ["C:/foo"], ["s3:///C:/foo"], 0),
        ("win32", ["file:///C:/foo"], ["s3:///C:/foo"], 0),
    ],
)
def test_scan_delta_dv_normalize_scheme(
    platform: str | None,
    requested_paths: list[str],
    dv_paths: list[str],
    n_matches: int,
) -> None:
    if platform == "win32" and sys.platform != "win32":
        pytest.skip("windows-only test")
    if platform == "posix" and sys.platform == "win32":
        pytest.skip("posix-only test")

    requested_df = pl.DataFrame({"path": requested_paths}, schema={"path": pl.String})
    delta_deletion_vectors = pl.DataFrame(
        {
            "filepath": dv_paths,
            "selection_vector": [[False] for _ in dv_paths],
        },
        schema={"filepath": pl.String, "selection_vector": pl.List(pl.Boolean)},
    )
    out = _extract_delta_deletion_vectors(requested_df, delta_deletion_vectors)
    out_non_null = out.select(pl.col("selection_vector").is_not_null().sum()).item()
    assert out_non_null == n_matches


#
# Test suite: delta with roaring bitmap DVs
#


@pytest.mark.slow
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
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    path = tmp_path / "delta_table"
    df = pl.DataFrame({"a": range(n_rows), "b": [f"b_{i}" for i in range(n_rows)]})
    data = df.to_arrow()
    create_dv_table(path, data, dv)

    out = pl.scan_delta(path).collect()
    capture = capfd.readouterr().err

    # Test: resulting df
    expected = df.with_row_index().filter(~pl.col.index.is_in(dv)).drop("index")
    assert_frame_equal(out, expected)

    # duckdb cross-check
    import duckdb

    conn = duckdb.connect()
    df_duckdb = conn.execute(f"SELECT * FROM delta_scan('{path}')").pl()
    assert_frame_equal(out, df_duckdb, check_row_order=False)

    # Test: py deletion_vectors() API contract
    dv_df = pl.DataFrame(DeltaTable(path).deletion_vectors())
    parquet_path = list(path.glob("*.parquet"))
    assert len(parquet_path) == 1

    # Since delta may truncate trailing trues, we normalize both
    # to truncated form for comparison.
    def truncate_trailing_trues(vec: list[bool]) -> list[bool]:
        v = list(vec)
        while v and v[-1]:
            v.pop()
        return v

    observed_vec = truncate_trailing_trues(dv_df["selection_vector"][0].to_list())
    expected_vec = truncate_trailing_trues([i not in dv for i in range(n_rows)])

    expected_dv_df = pl.DataFrame(
        {
            "filepath": [parquet_path[0].as_uri()],
            "selection_vector": [expected_vec],
        },
        schema_overrides={"selection_vector": pl.List(pl.Boolean)},
    )
    normalized_dv_df = dv_df.with_columns(
        pl.Series("selection_vector", [observed_vec], dtype=pl.List(pl.Boolean))
    )
    assert_frame_equal(normalized_dv_df, expected_dv_df)

    # Test: stderr feedback
    # TODO: known issue: no message is printed when the resulting df is empty
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


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.xfail(
    strict=True,
    reason="canary: file_uris() and deletion_vector() both url-encode paths",
)
def test_scan_delta_dv_percent_encoded_path_canary(tmp_path: Path) -> None:
    path = tmp_path / "file#1_delta"
    df = pl.DataFrame({"a": range(5)})
    create_dv_table(path, df.to_arrow(), deleted_rows=[1, 3])

    out = pl.scan_delta(str(path)).collect()
    expected = df.filter(~pl.col.a.is_in([1, 3]))
    assert_frame_equal(out, expected)


@pytest.mark.slow
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
) -> None:
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

    # duckdb cross-check
    import duckdb

    conn = duckdb.connect()
    df_duckdb = conn.execute(f"SELECT * FROM delta_scan('{path}')").pl()
    assert_frame_equal(out, df_duckdb, check_row_order=False)


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
    import duckdb

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

    # sample limits to include boundaries, see tightBounds above
    max_b = (n_files * n_rows - 1) * 10
    deleted_b_values = {dfs[i]["b"][j] for i, dv in enumerate(dvs) for j in dv}
    sample_limits = sorted({0, max_b // 2, max_b, max_b + 10, *deleted_b_values})

    for limit in sample_limits:
        expr = pl.col.b >= limit
        out = pl.scan_delta(path).filter(expr).collect()
        capture = capfd.readouterr().err

        # note: n_skip_files ignores the presence of deletion vectors
        # because statistics are not updated (tightBounds = False)
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

        # duckdb cross-check
        conn = duckdb.connect()
        df_duckdb = (
            conn.execute(f"SELECT * FROM delta_scan('{path}')").pl().filter(expr)
        )
        assert_frame_equal(out, df_duckdb, check_row_order=False)


#
# Test suite: parquet/delta with mock DVs
#


def _mock_deletion_vector_callback(
    paths: pl.DataFrame,
    n_rows: int,
    dvs: list[list[int]],
) -> pl.DataFrame:
    path_list = paths["path"].to_list()

    selection_vectors = [[i not in dv for i in range(n_rows)] for dv in dvs]

    result = _extract_delta_deletion_vectors(
        paths,
        pl.DataFrame(
            {
                "filepath": [Path(p).as_uri() for p in path_list],
                "selection_vector": selection_vectors,
            }
        ),
    )
    return result


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
def test_scan_delta_dv_from_parquet_mock(
    n_files: int,
    n_rows: int,
    dvs: list[list[int]],
    tmp_path: Path,
) -> None:
    dfs = []
    for i in range(n_files):
        start = i * n_rows
        df = pl.DataFrame(
            {
                "a": range(start, start + n_rows),
                "b": range(start * 10, (start + n_rows) * 10, 10),
                "file_idx": i,
            }
        )
        dfs.append(df)

    for i, df in enumerate(dfs):
        df.lazy().sink_parquet(tmp_path / f"df_{i}.parquet")

    paths = tmp_path / "*.parquet"
    dv_callback = functools.partial(
        _mock_deletion_vector_callback, n_rows=n_rows, dvs=dvs
    )

    # order is preserved in the case of parquet file-by-file
    out = pl.scan_parquet(
        paths,
        _deletion_files=("delta-deletion-vector", dv_callback),
    ).collect()

    expected = pl.concat(
        [
            df.with_row_index().filter(~pl.col.index.is_in(dv)).drop("index")
            for df, dv in zip(dfs, dvs, strict=True)
        ]
    )

    assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.write_disk
@pytest.mark.parametrize(
    ("n_rows", "dv", "head_n"),
    [
        (10, [1, 3, 5], 4),
        (10, [1, 3, 5], 0),
        (10, [1, 3, 5], 10),
        (10, [], 4),
        (10, list(range(10)), 4),
        (5, [0, 1], 2),
    ],
)
def test_scan_delta_dv_slice_mock(
    n_rows: int,
    dv: list[int],
    head_n: int,
    tmp_path: Path,
) -> None:
    df = pl.DataFrame({"a": range(n_rows), "b": [f"b_{i}" for i in range(n_rows)]})
    df.lazy().sink_parquet(tmp_path / "df_0.parquet")

    dv_callback = functools.partial(
        _mock_deletion_vector_callback, n_rows=n_rows, dvs=[dv]
    )

    out = (
        pl.scan_parquet(
            tmp_path / "*.parquet",
            _deletion_files=("delta-deletion-vector", dv_callback),
        )
        .head(head_n)
        .collect()
    )

    expected = (
        df.with_row_index().filter(~pl.col.index.is_in(dv)).drop("index").head(head_n)
    )

    assert_frame_equal(out, expected)


@pytest.mark.write_disk
@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_files", "n_rows", "dvs"),
    [
        (3, 5, [[1, 2, 3], [], [2]]),
        (3, 5, [[], [0, 1, 2], [2]]),
        (3, 5, [[], [], []]),
        (3, 5, [list(range(5)), list(range(5)), list(range(5))]),
    ],
)
def test_scan_delta_dv_delta_sink_mock(
    n_files: int,
    n_rows: int,
    dvs: list[list[int]],
    tmp_path: Path,
) -> None:
    dfs = []
    for i in range(n_files):
        start = i * n_rows
        df = pl.DataFrame(
            {
                "row": range(start, start + n_rows),
                "file": i,
            }
        )
        dfs.append(df)

    for df in dfs:
        df.lazy().sink_delta(tmp_path, mode="append")

    # note: delta has no order maintaining guarantees with respect to file or row (!?)
    # therefore: we track file and row mapping explicitl by index inside the dataframe,
    # and extract this from the file as written to stub the right deletion_vectors
    path_to_dv: dict[str, list[int]] = {}
    for p in tmp_path.glob("*.parquet"):
        file_idx = pl.read_parquet(p, columns=["file"])["file"][0]
        path_to_dv[str(p)] = dvs[file_idx]

    def _callback(paths: pl.DataFrame) -> pl.DataFrame:
        path_list = paths["path"].to_list()
        # row order within each file may differ from original df,
        # so look up deletions by actual 'a' value not position
        selection_vectors = []
        for p in path_list:
            # caveat - we are re-entering polars from within the callback
            file_data = pl.read_parquet(p, columns=["row", "file"])
            file_idx = file_data["file"][0]
            dv = dvs[file_idx]
            deleted_rows = set(dfs[file_idx]["row"].gather(dv).to_list())
            vec = file_data["row"].is_in(deleted_rows).not_().to_list()
            selection_vectors.append(vec)

        return _extract_delta_deletion_vectors(
            paths,
            pl.DataFrame(
                {
                    "filepath": [Path(p).as_uri() for p in path_list],
                    "selection_vector": selection_vectors,
                }
            ),
        )

    out = pl.scan_parquet(
        tmp_path / "*.parquet",
        _deletion_files=("delta-deletion-vector", _callback),
    ).collect()

    # expected: concat surviving rows from each df, order-independent
    expected = pl.concat(
        [
            df.with_row_index().filter(~pl.col.index.is_in(dv)).drop("index")
            for df, dv in zip(dfs, dvs, strict=True)
        ]
    )

    assert_frame_equal(
        out,
        expected,
        check_row_order=False,
    )


@pytest.mark.write_disk
def test_scan_delta_dv_requires_deltalake_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "delta_table"
    df = pl.DataFrame({"a": [1, 2, 3]})
    create_dv_table(path, df.to_arrow(), deleted_rows=[0])

    import deltalake

    monkeypatch.setattr(deltalake, "__version__", "1.4.1")

    with pytest.raises(ImportError, match=r"deltalake >= 1.4.2"):
        pl.scan_delta(path).collect()
