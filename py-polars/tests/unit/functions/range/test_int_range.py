from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_arange() -> None:
    ldf = pl.LazyFrame({"a": [1, 1, 1]})
    result = ldf.filter(pl.col("a") >= pl.arange(0, 3)).collect()
    expected = pl.DataFrame({"a": [1, 1]})
    assert_frame_equal(result, expected)


def test_int_range_decreasing() -> None:
    assert pl.int_range(10, 1, -2, eager=True).to_list() == list(range(10, 1, -2))
    assert pl.int_range(10, -1, -1, eager=True).to_list() == list(range(10, -1, -1))


def test_int_range_expr() -> None:
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    out = df.select(pl.int_range(0, pl.col("a").count() * 10))
    assert out.shape == (20, 1)
    assert out.to_series(0)[-1] == 19

    # eager arange
    out2 = pl.arange(0, 10, 2, eager=True)
    assert out2.to_list() == [0, 2, 4, 6, 8]


def test_int_range() -> None:
    result = pl.int_range(0, 3)
    expected = pl.Series("int", [0, 1, 2])
    assert_series_equal(pl.select(result).to_series(), expected)


def test_int_range_eager() -> None:
    result = pl.int_range(0, 3, eager=True)
    expected = pl.Series("int", [0, 1, 2])
    assert_series_equal(result, expected)


def test_int_range_schema() -> None:
    result = pl.LazyFrame().select(pl.int_range(-3, 3))

    expected_schema = {"int": pl.Int64}
    assert result.schema == expected_schema
    assert result.collect().schema == expected_schema


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("a", "b", pl.Series("int_range", [[1, 2], [2, 3]])),
        (-1, "a", pl.Series("int_range", [[-1, 0], [-1, 0, 1]])),
        ("b", 4, pl.Series("int_range", [[3], []])),
    ],
)
def test_int_ranges(start: Any, end: Any, expected: pl.Series) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

    result = df.select(pl.int_ranges(start, end))
    assert_series_equal(result.to_series(), expected)


def test_int_ranges_eager() -> None:
    start = pl.Series([1, 2])
    result = pl.int_ranges(start, 4, eager=True)

    expected = pl.Series("int_range", [[1, 2, 3], [2, 3]])
    assert_series_equal(result, expected)


def test_int_ranges_schema_dtype_default() -> None:
    lf = pl.LazyFrame({"start": [1, 2], "end": [3, 4]})

    result = lf.select(pl.int_ranges("start", "end"))

    expected_schema = {"int_range": pl.List(pl.Int64)}
    assert result.schema == expected_schema
    assert result.collect().schema == expected_schema


def test_int_ranges_schema_dtype_arg() -> None:
    lf = pl.LazyFrame({"start": [1, 2], "end": [3, 4]})

    result = lf.select(pl.int_ranges("start", "end", dtype=pl.UInt16))

    expected_schema = {"int_range": pl.List(pl.UInt16)}
    assert result.schema == expected_schema
    assert result.collect().schema == expected_schema


def test_int_range_input_shape_empty() -> None:
    empty = pl.Series(dtype=pl.Time)
    single = pl.Series([5])

    with pytest.raises(
        pl.ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.int_range(empty, single, eager=True)
    with pytest.raises(
        pl.ComputeError, match="`end` must contain exactly one value, got 0 values"
    ):
        pl.int_range(single, empty, eager=True)
    with pytest.raises(
        pl.ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.int_range(empty, empty, eager=True)


def test_int_range_input_shape_multiple_values() -> None:
    single = pl.Series([5])
    multiple = pl.Series([10, 15])

    with pytest.raises(
        pl.ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.int_range(multiple, single, eager=True)
    with pytest.raises(
        pl.ComputeError, match="`end` must contain exactly one value, got 2 values"
    ):
        pl.int_range(single, multiple, eager=True)
    with pytest.raises(
        pl.ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.int_range(multiple, multiple, eager=True)
