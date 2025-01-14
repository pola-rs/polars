from __future__ import annotations

import re
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
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
def test_linear_space(
    start: int | float,
    end: int | float,
    num_samples: int,
    interval: ClosedInterval,
    eager: bool,
) -> None:
    if eager:
        result = pl.select(
            pl.linear_space(start, end, num_samples, closed=interval).alias("ls")
        ).to_series()
    else:
        result = pl.linear_space(
            start, end, num_samples, closed=interval, eager=True
        ).rename("ls")

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
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    result = df.select(pl.linear_space(0, pl.col("a").len(), 3))
    expected = pl.DataFrame({"literal": pl.Series([0.0, 2.5, 5.0], dtype=pl.Float64)})
    assert_frame_equal(result, expected)

    result = df.select(pl.linear_space(pl.col("a").len(), 0, 3))
    expected = pl.DataFrame({"a": pl.Series([5.0, 2.5, 0.0], dtype=pl.Float64)})
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
    result = pl.linear_space(
        pl.lit(0, dtype=dtype_start),
        pl.lit(1, dtype=dtype_end),
        6,
        eager=True,
    )
    expected = pl.Series(
        "literal", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=dtype_expected
    )
    assert_series_equal(result, expected)


def test_linear_space_date() -> None:
    d1 = date(2025, 1, 1)
    d2 = date(2025, 2, 1)
    result = pl.linear_space(d1, d2, 4, eager=True)
    out_values = [
        datetime(2025, 1, 1),
        datetime(2025, 1, 11, 8),
        datetime(2025, 1, 21, 16),
        datetime(2025, 2, 1),
    ]

    result = pl.linear_space(d1, d2, 4, eager=True, closed="both")
    expected = pl.Series("literal", out_values, dtype=pl.Datetime("ms"))
    assert_series_equal(result, expected)

    result = pl.linear_space(d1, d2, 3, eager=True, closed="left")
    expected = pl.Series("literal", out_values[:-1], dtype=pl.Datetime("ms"))
    assert_series_equal(result, expected)

    result = pl.linear_space(d1, d2, 3, eager=True, closed="right")
    expected = pl.Series("literal", out_values[1:], dtype=pl.Datetime("ms"))
    assert_series_equal(result, expected)

    result = pl.linear_space(d1, d2, 2, eager=True, closed="none")
    expected = pl.Series("literal", out_values[1:-1], dtype=pl.Datetime("ms"))
    assert_series_equal(result, expected)


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

    result_int = pl.linear_space(start, end, 11, eager=True).cast(pl.Int64)
    result_dt = pl.linear_space(
        pl.lit(start, dtype=dtype), pl.lit(end, dtype=dtype), 11, eager=True
    ).cast(pl.Int64)

    assert_series_equal(result_int, result_dt)


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
def test_linear_space_incompatible_temporals(
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
