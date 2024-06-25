from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


def test_int_range() -> None:
    result = pl.int_range(0, 3)
    expected = pl.Series("int_range", [0, 1, 2])
    assert_series_equal(pl.select(int_range=result).to_series(), expected)


def test_int_range_alias() -> None:
    # note: `arange` is an alias for `int_range`
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


def test_int_range_short_syntax() -> None:
    result = pl.int_range(3)
    expected = pl.Series("int", [0, 1, 2])
    assert_series_equal(pl.select(int=result).to_series(), expected)


def test_int_ranges_short_syntax() -> None:
    result = pl.int_ranges(3)
    expected = pl.Series("int", [[0, 1, 2]])
    assert_series_equal(pl.select(int=result).to_series(), expected)


def test_int_range_start_default() -> None:
    result = pl.int_range(end=3)
    expected = pl.Series("int", [0, 1, 2])
    assert_series_equal(pl.select(int=result).to_series(), expected)


def test_int_ranges_start_default() -> None:
    df = pl.DataFrame({"end": [3, 2]})
    result = df.select(int_range=pl.int_ranges(end="end"))
    expected = pl.DataFrame({"int_range": [[0, 1, 2], [0, 1]]})
    assert_frame_equal(result, expected)


def test_int_range_eager() -> None:
    result = pl.int_range(0, 3, eager=True)
    expected = pl.Series("literal", [0, 1, 2])
    assert_series_equal(result, expected)


def test_int_range_schema() -> None:
    result = pl.LazyFrame().select(int=pl.int_range(-3, 3))

    expected_schema = {"int": pl.Int64}
    assert result.collect_schema() == expected_schema
    assert result.collect().schema == expected_schema


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("a", "b", pl.Series("a", [[1, 2], [2, 3]])),
        (-1, "a", pl.Series("literal", [[-1, 0], [-1, 0, 1]])),
        ("b", 4, pl.Series("b", [[3], []])),
    ],
)
def test_int_ranges(start: Any, end: Any, expected: pl.Series) -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

    result = df.select(pl.int_ranges(start, end))
    assert_series_equal(result.to_series(), expected)


def test_int_ranges_decreasing() -> None:
    expected = pl.Series("literal", [[5, 4, 3, 2, 1]], dtype=pl.List(pl.Int64))
    assert_series_equal(pl.int_ranges(5, 0, -1, eager=True), expected)
    assert_series_equal(pl.select(pl.int_ranges(5, 0, -1)).to_series(), expected)


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [
        (0, -5, 1),
        (5, 0, 1),
        (0, 5, -1),
    ],
)
def test_int_ranges_empty(start: int, end: int, step: int) -> None:
    assert_series_equal(
        pl.int_range(start, end, step, eager=True),
        pl.Series("literal", [], dtype=pl.Int64),
    )
    assert_series_equal(
        pl.int_ranges(start, end, step, eager=True),
        pl.Series("literal", [[]], dtype=pl.List(pl.Int64)),
    )
    assert_series_equal(
        pl.Series("int", [], dtype=pl.Int64),
        pl.select(int=pl.int_range(start, end, step)).to_series(),
    )
    assert_series_equal(
        pl.Series("int_range", [[]], dtype=pl.List(pl.Int64)),
        pl.select(int_range=pl.int_ranges(start, end, step)).to_series(),
    )


def test_int_ranges_eager() -> None:
    start = pl.Series("s", [1, 2])
    result = pl.int_ranges(start, 4, eager=True)

    expected = pl.Series("s", [[1, 2, 3], [2, 3]])
    assert_series_equal(result, expected)


def test_int_ranges_schema_dtype_default() -> None:
    lf = pl.LazyFrame({"start": [1, 2], "end": [3, 4]})

    result = lf.select(pl.int_ranges("start", "end"))

    expected_schema = {"start": pl.List(pl.Int64)}
    assert result.collect_schema() == expected_schema
    assert result.collect().schema == expected_schema


def test_int_ranges_schema_dtype_arg() -> None:
    lf = pl.LazyFrame({"start": [1, 2], "end": [3, 4]})

    result = lf.select(pl.int_ranges("start", "end", dtype=pl.UInt16))

    expected_schema = {"start": pl.List(pl.UInt16)}
    assert result.collect_schema() == expected_schema
    assert result.collect().schema == expected_schema


def test_int_range_input_shape_empty() -> None:
    empty = pl.Series(dtype=pl.Time)
    single = pl.Series([5])

    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.int_range(empty, single, eager=True)
    with pytest.raises(
        ComputeError, match="`end` must contain exactly one value, got 0 values"
    ):
        pl.int_range(single, empty, eager=True)
    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 0 values"
    ):
        pl.int_range(empty, empty, eager=True)


def test_int_range_input_shape_multiple_values() -> None:
    single = pl.Series([5])
    multiple = pl.Series([10, 15])

    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.int_range(multiple, single, eager=True)
    with pytest.raises(
        ComputeError, match="`end` must contain exactly one value, got 2 values"
    ):
        pl.int_range(single, multiple, eager=True)
    with pytest.raises(
        ComputeError, match="`start` must contain exactly one value, got 2 values"
    ):
        pl.int_range(multiple, multiple, eager=True)


# https://github.com/pola-rs/polars/issues/10867
def test_int_range_index_type_negative() -> None:
    result = pl.select(pl.int_range(pl.lit(3).cast(pl.UInt32).alias("start"), -1, -1))
    expected = pl.DataFrame({"start": [3, 2, 1, 0]})
    assert_frame_equal(result, expected)


def test_int_range_null_input() -> None:
    with pytest.raises(ComputeError, match="invalid null input for `int_range`"):
        pl.select(pl.int_range(3, pl.lit(None), -1, dtype=pl.UInt32))


def test_int_range_invalid_conversion() -> None:
    with pytest.raises(
        InvalidOperationError, match="conversion from `i32` to `u32` failed"
    ):
        pl.select(pl.int_range(3, -1, -1, dtype=pl.UInt32))


def test_int_range_non_integer_dtype() -> None:
    with pytest.raises(
        ComputeError, match="non-integer `dtype` passed to `int_range`: Float64"
    ):
        pl.select(pl.int_range(3, -1, -1, dtype=pl.Float64))  # type: ignore[arg-type]


def test_int_ranges_broadcasting() -> None:
    df = pl.DataFrame({"int": [1, 2, 3]})
    result = df.select(
        # result column name means these columns will be broadcast
        pl.int_ranges(1, pl.Series([2, 4, 6]), "int").alias("start"),
        pl.int_ranges("int", 6, "int").alias("end"),
        pl.int_ranges("int", pl.col("int") + 2, 1).alias("step"),
        pl.int_ranges("int", 3, 1).alias("end_step"),
        pl.int_ranges(1, "int", 1).alias("start_step"),
        pl.int_ranges(1, 6, "int").alias("start_end"),
        pl.int_ranges("int", pl.Series([4, 5, 10]), "int").alias("no_broadcast"),
    )
    expected = pl.DataFrame(
        {
            "start": [[1], [1, 3], [1, 4]],
            "end": [
                [1, 2, 3, 4, 5],
                [2, 4],
                [3],
            ],
            "step": [[1, 2], [2, 3], [3, 4]],
            "end_step": [
                [1, 2],
                [2],
                [],
            ],
            "start_step": [
                [],
                [1],
                [1, 2],
            ],
            "start_end": [
                [1, 2, 3, 4, 5],
                [1, 3, 5],
                [1, 4],
            ],
            "no_broadcast": [[1, 2, 3], [2, 4], [3, 6, 9]],
        }
    )
    assert_frame_equal(result, expected)


# https://github.com/pola-rs/polars/issues/15307
def test_int_range_non_int_dtype() -> None:
    with pytest.raises(
        ComputeError, match="non-integer `dtype` passed to `int_range`: String"
    ):
        pl.int_range(0, 3, dtype=pl.String, eager=True)  # type: ignore[arg-type]


# https://github.com/pola-rs/polars/issues/15307
def test_int_ranges_non_int_dtype() -> None:
    with pytest.raises(
        ComputeError, match="non-integer `dtype` passed to `int_ranges`: String"
    ):
        pl.int_ranges(0, 3, dtype=pl.String, eager=True)  # type: ignore[arg-type]
