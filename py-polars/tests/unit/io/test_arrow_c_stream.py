"""Tests for Arrow C Stream scanning functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType

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


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.limit(3).filter(pl.col("a") > 2),
        lambda lf: lf.head(3).filter(pl.col("a") > 2),
        lambda lf: lf.slice(0, 3).filter(pl.col("a") > 2),
        lambda lf: lf.limit(2).filter(pl.col("a") > 2),
        lambda lf: lf.limit(4).filter(pl.col("a") % 2 == 0),
        lambda lf: lf.limit(3).filter(pl.col("a") > 2).select("b"),
        lambda lf: lf.limit(2).filter(pl.col("a") > 2).limit(1),
        lambda lf: lf.filter(pl.col("a") > 2).limit(2),
    ],
)
def test_scan_arrow_c_stream_slice_before_predicate_30916(
    query: Any, engine: EngineType
) -> None:
    # A pushed-down slice counts source rows, before the pushed-down predicate.
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": ["x", "y", "z", "a", "b", "c"],
            "c": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    table = df.to_arrow()
    reader = pa.RecordBatchReader.from_batches(
        table.schema, table.to_batches(max_chunksize=2)
    )

    out = query(pl.scan_arrow_c_stream(reader)).collect(engine=engine)
    assert_frame_equal(out, query(df.lazy()).collect())


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
