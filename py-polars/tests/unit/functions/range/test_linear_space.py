from __future__ import annotations

import re
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars import Expr
    from polars._typing import ClosedInterval, PolarsDataType


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (0, 0),
        (0, 1),
        (-1, 0),
        (-2.1, 3.4),
    ],
)
@pytest.mark.parametrize("num_samples", [0, 1, 2, 5, 1_000])
@pytest.mark.parametrize("interval", ["both", "left", "right", "none"])
@pytest.mark.parametrize("eager", [True, False])
def test_linear_space_values(
    start: int | float,
    end: int | float,
    num_samples: int,
    interval: ClosedInterval,
    eager: bool,
) -> None:
    if eager:
        result = pl.linear_space(
            start, end, num_samples, closed=interval, eager=True
        ).rename("ls")
    else:
        result = pl.select(
            ls=pl.linear_space(start, end, num_samples, closed=interval)
        ).to_series()

    if interval == "both":
        expected = pl.Series("ls", np.linspace(start, end, num_samples))
    elif interval == "left":
        expected = pl.Series("ls", np.linspace(start, end, num_samples, endpoint=False))
    elif interval == "right":
        expected = pl.Series("ls", np.linspace(start, end, num_samples + 1)[1:])
    elif interval == "none":
        expected = pl.Series("ls", np.linspace(start, end, num_samples + 2)[1:-1])

    assert_series_equal(result, expected)


def test_linear_space_expr() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})

    result = lf.select(pl.linear_space(0, pl.col("a").len(), 3))
    expected = lf.select(literal=pl.Series([0.0, 2.5, 5.0], dtype=pl.Float64))
    assert_frame_equal(result, expected)

    result = lf.select(pl.linear_space(pl.col("a").len(), 0, 3))
    expected = lf.select(a=pl.Series([5.0, 2.5, 0.0], dtype=pl.Float64))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("dtype_start", "dtype_end", "dtype_expected"),
    [
        (pl.Float32, pl.Float32, pl.Float32),
        (pl.Float32, pl.Float64, pl.Float64),
        (pl.Float64, pl.Float32, pl.Float64),
        (pl.Float64, pl.Float64, pl.Float64),
        (pl.UInt8, pl.UInt32, pl.Float64),
        (pl.Int16, pl.Int128, pl.Float64),
        (pl.Int8, pl.Float64, pl.Float64),
    ],
)
def test_linear_space_numeric_dtype(
    dtype_start: PolarsDataType,
    dtype_end: PolarsDataType,
    dtype_expected: PolarsDataType,
) -> None:
    lf = pl.LazyFrame()
    result = lf.select(
        ls=pl.linear_space(pl.lit(0, dtype=dtype_start), pl.lit(1, dtype=dtype_end), 6)
    )
    expected = lf.select(
        ls=pl.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=dtype_expected)
    )
    assert_frame_equal(result, expected)


def test_linear_space_date() -> None:
    d1 = date(2025, 1, 1)
    d2 = date(2025, 2, 1)
    out_values = [
        datetime(2025, 1, 1),
        datetime(2025, 1, 11, 8),
        datetime(2025, 1, 21, 16),
        datetime(2025, 2, 1),
    ]
    lf = pl.LazyFrame()

    result = lf.select(ls=pl.linear_space(d1, d2, 4, closed="both"))
    expected = lf.select(ls=pl.Series(out_values, dtype=pl.Datetime("ms")))
    assert_frame_equal(result, expected)

    result = lf.select(ls=pl.linear_space(d1, d2, 3, closed="left"))
    expected = lf.select(ls=pl.Series(out_values[:-1], dtype=pl.Datetime("ms")))
    assert_frame_equal(result, expected)

    result = lf.select(ls=pl.linear_space(d1, d2, 3, closed="right"))
    expected = lf.select(ls=pl.Series(out_values[1:], dtype=pl.Datetime("ms")))
    assert_frame_equal(result, expected)

    result = lf.select(ls=pl.linear_space(d1, d2, 2, closed="none"))
    expected = lf.select(ls=pl.Series(out_values[1:-1], dtype=pl.Datetime("ms")))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Datetime("ms", None),
        pl.Datetime("ms", time_zone="Asia/Tokyo"),
        pl.Datetime("us", None),
        pl.Datetime("us", time_zone="Asia/Tokyo"),
        pl.Datetime("ns", time_zone="Asia/Tokyo"),
        pl.Time,
        pl.Duration("ms"),
        pl.Duration("us"),
        pl.Duration("ns"),
    ],
)
def test_linear_space_temporal(dtype: PolarsDataType) -> None:
    # All temporal types except for Date, which is tested above.
    start = 0
    end = 1_000_000_000

    lf = pl.LazyFrame()

    result_int = lf.select(
        ls=pl.linear_space(start, end, 11).cast(pl.Int64).cast(dtype)
    )
    result_dt = lf.select(
        ls=pl.linear_space(pl.lit(start, dtype=dtype), pl.lit(end, dtype=dtype), 11)
    )

    assert_frame_equal(result_int, result_dt)


@pytest.mark.parametrize(
    ("dtype1", "dtype2", "str1", "str2"),
    [
        (pl.Date, pl.Datetime("ms"), "Date", "Datetime(Milliseconds, None)"),
        (
            pl.Datetime("ms"),
            pl.Datetime("ns"),
            "Datetime(Milliseconds, None)",
            "Datetime(Nanoseconds, None)",
        ),
        (pl.Datetime("us"), pl.Time, "Datetime(Microseconds, None)", "Time"),
        (
            pl.Duration("us"),
            pl.Duration("ms"),
            "Duration(Microseconds)",
            "Duration(Milliseconds)",
        ),
        (pl.Int32, pl.String, "Int32", "String"),
    ],
)
def test_linear_space_incompatible_dtypes(
    dtype1: PolarsDataType,
    dtype2: PolarsDataType,
    str1: str,
    str2: str,
) -> None:
    value1 = pl.lit(0, dtype1)
    value2 = pl.lit(1, dtype2)
    with pytest.raises(
        ComputeError,
        match=re.escape(
            f"'start' and 'end' have incompatible dtypes, got {str1} and {str2}"
        ),
    ):
        pl.linear_space(value1, value2, 11, eager=True)


def test_linear_space_expr_wrong_length() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    msg = "unable to add a column of length 6 to a DataFrame of height 5"
    streaming_msg = "zip node received non-equal length inputs"
    with pytest.raises(ShapeError, match=rf"({msg})|({streaming_msg})"):
        df.with_columns(pl.linear_space(0, 1, 6))


def test_linear_space_num_samples_expr() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    result = lf.with_columns(ls=pl.linear_space(0, 1, pl.len(), closed="left"))
    expected = lf.with_columns(ls=pl.Series([0, 0.2, 0.4, 0.6, 0.8], dtype=pl.Float64))
    assert_frame_equal(result, expected)


def test_linear_space_invalid_num_samples_expr() -> None:
    lf = pl.LazyFrame({"x": [1, 2, 3]})
    with pytest.raises(
        ComputeError, match="`num_samples` must contain exactly one value, got 3 values"
    ):
        lf.select(pl.linear_space(0, 1, pl.col("x"))).collect()


@pytest.mark.parametrize("interval", ["both", "left", "right", "none"])
def test_linear_spaces_values(interval: ClosedInterval) -> None:
    starts = [
        None, 0.0, 0.0, 0.0, 0.0,
        0.0, None, 0.0, 0.0, 0.0,
        -1.0, -1.0, None, -1.0, -1.0,
        -2.1, -2.1, -2.1, None, -2.1,
    ]  # fmt: skip

    ends = [
        0.0, None, 0.0, 0.0, 0.0,
        1.0, 1.0, None, 1.0, 1.0,
        0.0, 0.0, 0.0, None, 0.0,
        3.4, 3.4, 3.4, 3.4, None,
    ]  # fmt: skip

    num_samples = [
        0, 1, None, 5, 1_1000,
        0, 1, 2, 5, None,
        0, 1, 2, 5, 1_1000,
        0, 1, 2, 5, 1_1000,
    ]  # fmt: skip

    df = pl.DataFrame(
        {
            "start": starts,
            "end": ends,
            "num_samples": num_samples,
        }
    )

    out = df.select(pl.linear_spaces("start", "end", "num_samples", closed=interval))[
        "start"
    ]

    # We check each element against the output from pl.linear_space(), which is
    # validated above.
    for row, start, end, ns in zip(out, starts, ends, num_samples):
        if start is None or end is None or ns is None:
            assert row is None
        else:
            expected = pl.linear_space(
                start, end, ns, eager=True, closed=interval
            ).rename("")
            assert_series_equal(row, expected)


@pytest.mark.parametrize("interval", ["both", "left", "right", "none"])
def test_linear_spaces_one_numeric(interval: ClosedInterval) -> None:
    # Two expressions, one numeric input
    starts = [1, 2]
    ends = [5, 6]
    num_samples = [3, 4]
    lf = pl.LazyFrame(
        {
            "start": starts,
            "end": ends,
            "num_samples": num_samples,
        }
    )
    result = lf.select(
        pl.linear_spaces(starts[0], "end", "num_samples", closed=interval).alias(
            "start"
        ),
        pl.linear_spaces("start", ends[0], "num_samples", closed=interval).alias("end"),
        pl.linear_spaces("start", "end", num_samples[0], closed=interval).alias(
            "num_samples"
        ),
    )
    expected_start0 = pl.linear_space(
        starts[0], ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_start1 = pl.linear_space(
        starts[0], ends[1], num_samples[1], closed=interval, eager=True
    )
    expected_end0 = pl.linear_space(
        starts[0], ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_end1 = pl.linear_space(
        starts[1], ends[0], num_samples[1], closed=interval, eager=True
    )
    expected_ns0 = pl.linear_space(
        starts[0], ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_ns1 = pl.linear_space(
        starts[1], ends[1], num_samples[0], closed=interval, eager=True
    )
    expected = pl.LazyFrame(
        {
            "start": [expected_start0, expected_start1],
            "end": [expected_end0, expected_end1],
            "num_samples": [expected_ns0, expected_ns1],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("interval", ["both", "left", "right", "none"])
def test_linear_spaces_two_numeric(interval: ClosedInterval) -> None:
    # One expression, two numeric inputs
    starts = [1, 2]
    ends = [5, 6]
    num_samples = [3, 4]
    lf = pl.LazyFrame(
        {
            "start": starts,
            "end": ends,
            "num_samples": num_samples,
        }
    )
    result = lf.select(
        pl.linear_spaces("start", ends[0], num_samples[0], closed=interval).alias(
            "start"
        ),
        pl.linear_spaces(starts[0], "end", num_samples[0], closed=interval).alias(
            "end"
        ),
        pl.linear_spaces(starts[0], ends[0], "num_samples", closed=interval).alias(
            "num_samples"
        ),
    )
    expected_start0 = pl.linear_space(
        starts[0], ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_start1 = pl.linear_space(
        starts[1], ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_end0 = pl.linear_space(
        starts[0], ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_end1 = pl.linear_space(
        starts[0], ends[1], num_samples[0], closed=interval, eager=True
    )
    expected_ns0 = pl.linear_space(
        starts[0], ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_ns1 = pl.linear_space(
        starts[0], ends[0], num_samples[1], closed=interval, eager=True
    )
    expected = pl.LazyFrame(
        {
            "start": [expected_start0, expected_start1],
            "end": [expected_end0, expected_end1],
            "num_samples": [expected_ns0, expected_ns1],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "num_samples",
    [
        5,
        pl.lit(5),
        pl.lit(5, dtype=pl.UInt8),
        pl.lit(5, dtype=pl.UInt16),
        pl.lit(5, dtype=pl.UInt32),
        pl.lit(5, dtype=pl.UInt64),
        pl.lit(5, dtype=pl.Int8),
        pl.lit(5, dtype=pl.Int16),
        pl.lit(5, dtype=pl.Int32),
        pl.lit(5, dtype=pl.Int64),
    ],
)
@pytest.mark.parametrize("interval", ["both", "left", "right", "none"])
@pytest.mark.parametrize(
    "dtype",
    [
        pl.Float32,
        pl.Float64,
        pl.Datetime,
    ],
)
def test_linear_spaces_as_array(
    interval: ClosedInterval,
    num_samples: int | Expr,
    dtype: PolarsDataType,
) -> None:
    starts = [1, 2]
    ends = [5, 6]
    lf = pl.LazyFrame(
        {
            "start": pl.Series(starts, dtype=dtype),
            "end": pl.Series(ends, dtype=dtype),
        }
    )
    result = lf.select(
        a=pl.linear_spaces("start", "end", num_samples, closed=interval, as_array=True)
    )
    expected_0 = pl.linear_space(
        pl.lit(starts[0], dtype=dtype),
        pl.lit(ends[0], dtype=dtype),
        num_samples,
        closed=interval,
        eager=True,
    )
    expected_1 = pl.linear_space(
        pl.lit(starts[1], dtype=dtype),
        pl.lit(ends[1], dtype=dtype),
        num_samples,
        closed=interval,
        eager=True,
    )
    expected = pl.LazyFrame(
        {"a": pl.Series([expected_0, expected_1], dtype=pl.Array(dtype, 5))}
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("bad_num_samples", [pl.lit("a"), 1.0, "num_samples"])
def test_linear_space_invalid_as_array(bad_num_samples: Any) -> None:
    lf = pl.LazyFrame(
        {
            "start": [1, 2],
            "end": [5, 6],
            "num_samples": [2, 4],
        }
    )
    with pytest.raises(
        InvalidOperationError,
        match="'as_array' is only valid when 'num_samples' is a constant integer",
    ):
        lf.select(pl.linear_spaces("starts", "ends", bad_num_samples, as_array=True))


@pytest.mark.parametrize("interval", ["both", "left", "right", "none"])
def test_linear_spaces_numeric_input(interval: ClosedInterval) -> None:
    starts = [1, 2]
    ends = [5, 6]
    num_samples = [3, 4]
    lf = pl.LazyFrame(
        {
            "start": starts,
            "end": ends,
            "num_samples": num_samples,
        }
    )
    result = lf.select(
        pl.linear_spaces("start", "end", "num_samples", closed=interval).alias("all"),
        pl.linear_spaces(0, "end", "num_samples", closed=interval).alias("start"),
        pl.linear_spaces("start", 10, "num_samples", closed=interval).alias("end"),
        pl.linear_spaces("start", "end", 8, closed=interval).alias("num_samples"),
    )
    expected_all0 = pl.linear_space(
        starts[0], ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_all1 = pl.linear_space(
        starts[1], ends[1], num_samples[1], closed=interval, eager=True
    )
    expected_start0 = pl.linear_space(
        0, ends[0], num_samples[0], closed=interval, eager=True
    )
    expected_start1 = pl.linear_space(
        0, ends[1], num_samples[1], closed=interval, eager=True
    )
    expected_end0 = pl.linear_space(
        starts[0], 10, num_samples[0], closed=interval, eager=True
    )
    expected_end1 = pl.linear_space(
        starts[1], 10, num_samples[1], closed=interval, eager=True
    )
    expected_ns0 = pl.linear_space(starts[0], ends[0], 8, closed=interval, eager=True)
    expected_ns1 = pl.linear_space(starts[1], ends[1], 8, closed=interval, eager=True)
    expected = pl.LazyFrame(
        {
            "all": [expected_all0, expected_all1],
            "start": [expected_start0, expected_start1],
            "end": [expected_end0, expected_end1],
            "num_samples": [expected_ns0, expected_ns1],
        }
    )
    assert_frame_equal(result, expected)


def test_linear_spaces_date() -> None:
    d1 = date(2025, 1, 1)
    d2 = date(2025, 2, 1)

    lf = pl.LazyFrame(
        {
            "start": [None, d1, d1, d1, None, d1, d1, d1],
            "end": [d2, None, d2, d2, d2, None, d2, d2],
            "num_samples": [3, 3, None, 3, 4, 4, None, 4],
        }
    )

    result = lf.select(pl.linear_spaces("start", "end", "num_samples"))
    expected = pl.LazyFrame(
        {
            "start": pl.Series(
                [
                    None,
                    None,
                    None,
                    [
                        datetime(2025, 1, 1),
                        datetime(2025, 1, 16, 12),
                        datetime(2025, 2, 1),
                    ],
                    None,
                    None,
                    None,
                    [
                        datetime(2025, 1, 1),
                        datetime(2025, 1, 11, 8),
                        datetime(2025, 1, 21, 16),
                        datetime(2025, 2, 1),
                    ],
                ],
                dtype=pl.List(pl.Datetime(time_unit="ms")),
            )
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Datetime("ms", None),
        pl.Datetime("ms", time_zone="Asia/Tokyo"),
        pl.Datetime("us", None),
        pl.Datetime("us", time_zone="Asia/Tokyo"),
        pl.Datetime("ns", time_zone="Asia/Tokyo"),
        pl.Time,
        pl.Duration("ms"),
        pl.Duration("us"),
        pl.Duration("ns"),
    ],
)
def test_linear_spaces_temporal(dtype: PolarsDataType) -> None:
    # All temporal types except for Date, which is tested above.
    start = 0
    end = 1_000_000_000

    lf = pl.LazyFrame(
        {
            "start": [start, start],
            "end": [end, end],
            "num_samples": [10, 15],
        }
    )
    lf_temporal = lf.select(pl.col("start", "end").cast(dtype), "num_samples")
    result_int = lf.select(pl.linear_spaces("start", "end", "num_samples")).select(
        pl.col("start").cast(pl.List(dtype))
    )
    result_dt = lf_temporal.select(pl.linear_spaces("start", "end", "num_samples"))

    assert_frame_equal(result_int, result_dt)


@pytest.mark.parametrize(
    ("dtype1", "dtype2", "str1", "str2"),
    [
        (pl.Date, pl.Datetime("ms"), "Date", "Datetime(Milliseconds, None)"),
        (
            pl.Datetime("ms"),
            pl.Datetime("ns"),
            "Datetime(Milliseconds, None)",
            "Datetime(Nanoseconds, None)",
        ),
        (pl.Datetime("us"), pl.Time, "Datetime(Microseconds, None)", "Time"),
        (
            pl.Duration("us"),
            pl.Duration("ms"),
            "Duration(Microseconds)",
            "Duration(Milliseconds)",
        ),
        (pl.Int32, pl.String, "Int32", "String"),
    ],
)
def test_linear_spaces_incompatible_dtypes(
    dtype1: PolarsDataType,
    dtype2: PolarsDataType,
    str1: str,
    str2: str,
) -> None:
    df = pl.LazyFrame(
        {
            "start": pl.Series([0]).cast(dtype1),
            "end": pl.Series([1]).cast(dtype2),
            "num_samples": 3,
        }
    )
    with pytest.raises(
        ComputeError,
        match=re.escape(
            f"'start' and 'end' have incompatible dtypes, got {str1} and {str2}"
        ),
    ):
        df.select(pl.linear_spaces("start", "end", "num_samples")).collect()


def test_linear_spaces_f32() -> None:
    df = pl.LazyFrame(
        {
            "start": pl.Series([0.0, 1.0], dtype=pl.Float32),
            "end": pl.Series([10.0, 11.0], dtype=pl.Float32),
        }
    )
    result = df.select(pl.linear_spaces("start", "end", 6))
    expected = pl.LazyFrame(
        {
            "start": pl.Series(
                [
                    [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                    [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                ],
                dtype=pl.List(pl.Float32),
            )
        }
    )
    assert_frame_equal(result, expected)


def test_linear_spaces_eager() -> None:
    start = pl.Series("s", [1, 2])
    result = pl.linear_spaces(start, 6, 3, eager=True)

    expected = pl.Series("s", [[1.0, 3.5, 6.0], [2.0, 4.0, 6.0]])
    assert_series_equal(result, expected)
