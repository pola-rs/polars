"""Tests for Arrow C Stream scanning functionality."""

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

pa = pytest.importorskip("pyarrow")


def test_scan_arrow_c_stream_basic() -> None:
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    batches = [
        pa.record_batch([[1, 2, 3], ["x", "y", "z"]], schema=schema),
        pa.record_batch([[4, 5], ["a", "b"]], schema=schema),
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    lf = pl.scan_arrow_c_stream(reader)
    df = lf.collect()

    expected = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "a", "b"]})
    assert_frame_equal(df, expected)


def test_scan_arrow_c_stream_projection() -> None:
    schema = pa.schema([("a", pa.int64()), ("b", pa.string()), ("c", pa.float64())])
    batches = [pa.record_batch([[1, 2], ["x", "y"], [1.0, 2.0]], schema=schema)]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    lf = pl.scan_arrow_c_stream(reader)
    df = lf.select("a", "c").collect()

    assert df.columns == ["a", "c"]
    assert df.shape == (2, 2)


def test_scan_arrow_c_stream_filter() -> None:
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    batches = [
        pa.record_batch([[1, 2, 3, 4, 5], ["x", "y", "z", "a", "b"]], schema=schema)
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    lf = pl.scan_arrow_c_stream(reader)
    df = lf.filter(pl.col("a") > 2).collect()

    expected = pl.DataFrame({"a": [3, 4, 5], "b": ["z", "a", "b"]})
    assert_frame_equal(df, expected)


def test_scan_arrow_c_stream_n_rows_spans_batches() -> None:
    schema = pa.schema([("a", pa.int64())])
    batches = [
        pa.record_batch([[1, 2, 3]], schema=schema),
        pa.record_batch([[4, 5]], schema=schema),
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    lf = pl.scan_arrow_c_stream(reader)
    df = lf.head(4).collect()

    assert df["a"].to_list() == [1, 2, 3, 4]


def test_scan_arrow_c_stream_empty() -> None:
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    batches: list = []  # type: ignore[type-arg]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    lf = pl.scan_arrow_c_stream(reader)
    df = lf.collect()

    assert df.shape == (0, 2)
    assert df.columns == ["a", "b"]


def test_scan_arrow_c_stream_invalid_source() -> None:
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        pl.scan_arrow_c_stream("not a stream source")

    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        pl.scan_arrow_c_stream({"a": [1, 2, 3]})


def test_scan_arrow_c_stream_with_operations() -> None:
    schema = pa.schema([("name", pa.string()), ("value", pa.int64())])
    batches = [
        pa.record_batch([["alice", "bob", "charlie"], [10, 20, 30]], schema=schema)
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    lf = pl.scan_arrow_c_stream(reader)

    df = lf.with_columns((pl.col("value") * 2).alias("doubled")).collect()
    assert df.columns == ["name", "value", "doubled"]
    assert df["doubled"].to_list() == [20, 40, 60]


@pytest.mark.slow
def test_scan_arrow_c_stream_streaming_engine() -> None:
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    batches = [
        pa.record_batch([[1, 2, 3], ["x", "y", "z"]], schema=schema),
        pa.record_batch([[4, 5], ["a", "b"]], schema=schema),
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    lf = pl.scan_arrow_c_stream(reader)
    df = lf.collect(engine="streaming")

    expected = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "a", "b"]})
    assert_frame_equal(df, expected)


def test_scan_arrow_c_stream_multiple_dtypes() -> None:
    schema = pa.schema(
        [
            ("int_col", pa.int64()),
            ("float_col", pa.float64()),
            ("str_col", pa.string()),
            ("bool_col", pa.bool_()),
        ]
    )
    batches = [
        pa.record_batch(
            [
                [1, 2, 3],
                [1.1, 2.2, 3.3],
                ["a", "b", "c"],
                [True, False, True],
            ],
            schema=schema,
        )
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    lf = pl.scan_arrow_c_stream(reader)
    df = lf.collect()

    assert df.schema == {
        "int_col": pl.Int64,
        "float_col": pl.Float64,
        "str_col": pl.String,
        "bool_col": pl.Boolean,
    }
