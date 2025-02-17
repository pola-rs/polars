from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit


def test_replace_expr_datetime() -> None:
    df = pl.DataFrame(
        {
            "dates": [
                datetime(2088, 8, 8, 8, 8, 8, 8),
                datetime(2088, 8, 8, 8, 8, 8, 8),
                datetime(2088, 8, 8, 8, 8, 8, 8),
                datetime(2088, 8, 8, 8, 8, 8, 8),
                datetime(2088, 8, 8, 8, 8, 8, 8),
                datetime(2088, 8, 8, 8, 8, 8, 8),
                datetime(2088, 8, 8, 8, 8, 8, 8),
                None,
            ],
            "year": [None, 2, 3, 4, 5, 6, 7, 8],
            "month": [1, None, 3, 4, 5, 6, 7, 8],
            "day": [1, 2, None, 4, 5, 6, 7, 8],
            "hour": [1, 2, 3, None, 5, 6, 7, 8],
            "minute": [1, 2, 3, 4, None, 6, 7, 8],
            "second": [1, 2, 3, 4, 5, None, 7, 8],
            "microsecond": [1, 2, 3, 4, 5, 6, None, 8],
        }
    )

    result = df.select(
        pl.col("dates").dt.replace(
            year="year",
            month="month",
            day="day",
            hour="hour",
            minute="minute",
            second="second",
            microsecond="microsecond",
        )
    )

    expected = pl.DataFrame(
        {
            "dates": [
                datetime(2088, 1, 1, 1, 1, 1, 1),
                datetime(2, 8, 2, 2, 2, 2, 2),
                datetime(3, 3, 8, 3, 3, 3, 3),
                datetime(4, 4, 4, 8, 4, 4, 4),
                datetime(5, 5, 5, 5, 8, 5, 5),
                datetime(6, 6, 6, 6, 6, 8, 6),
                datetime(7, 7, 7, 7, 7, 7, 8),
                None,
            ]
        }
    )

    assert_frame_equal(result, expected)


def test_replace_expr_date() -> None:
    df = pl.DataFrame(
        {
            "dates": [date(2088, 8, 8), date(2088, 8, 8), date(2088, 8, 8), None],
            "year": [None, 2, 3, 4],
            "month": [1, None, 3, 4],
            "day": [1, 2, None, 4],
        }
    )

    result = df.select(
        pl.col("dates").dt.replace(year="year", month="month", day="day")
    )

    expected = pl.DataFrame(
        {"dates": [date(2088, 1, 1), date(2, 8, 2), date(3, 3, 8), None]}
    )

    assert_frame_equal(result, expected)


def test_replace_int_datetime() -> None:
    df = pl.DataFrame(
        {
            "a": [
                datetime(1, 1, 1, 1, 1, 1, 1),
                datetime(2, 2, 2, 2, 2, 2, 2),
                datetime(3, 3, 3, 3, 3, 3, 3),
                None,
            ]
        }
    )
    result = df.select(
        pl.col("a").dt.replace().alias("no_change"),
        pl.col("a").dt.replace(year=9).alias("year"),
        pl.col("a").dt.replace(month=9).alias("month"),
        pl.col("a").dt.replace(day=9).alias("day"),
        pl.col("a").dt.replace(hour=9).alias("hour"),
        pl.col("a").dt.replace(minute=9).alias("minute"),
        pl.col("a").dt.replace(second=9).alias("second"),
        pl.col("a").dt.replace(microsecond=9).alias("microsecond"),
    )
    expected = pl.DataFrame(
        {
            "no_change": [
                datetime(1, 1, 1, 1, 1, 1, 1),
                datetime(2, 2, 2, 2, 2, 2, 2),
                datetime(3, 3, 3, 3, 3, 3, 3),
                None,
            ],
            "year": [
                datetime(9, 1, 1, 1, 1, 1, 1),
                datetime(9, 2, 2, 2, 2, 2, 2),
                datetime(9, 3, 3, 3, 3, 3, 3),
                None,
            ],
            "month": [
                datetime(1, 9, 1, 1, 1, 1, 1),
                datetime(2, 9, 2, 2, 2, 2, 2),
                datetime(3, 9, 3, 3, 3, 3, 3),
                None,
            ],
            "day": [
                datetime(1, 1, 9, 1, 1, 1, 1),
                datetime(2, 2, 9, 2, 2, 2, 2),
                datetime(3, 3, 9, 3, 3, 3, 3),
                None,
            ],
            "hour": [
                datetime(1, 1, 1, 9, 1, 1, 1),
                datetime(2, 2, 2, 9, 2, 2, 2),
                datetime(3, 3, 3, 9, 3, 3, 3),
                None,
            ],
            "minute": [
                datetime(1, 1, 1, 1, 9, 1, 1),
                datetime(2, 2, 2, 2, 9, 2, 2),
                datetime(3, 3, 3, 3, 9, 3, 3),
                None,
            ],
            "second": [
                datetime(1, 1, 1, 1, 1, 9, 1),
                datetime(2, 2, 2, 2, 2, 9, 2),
                datetime(3, 3, 3, 3, 3, 9, 3),
                None,
            ],
            "microsecond": [
                datetime(1, 1, 1, 1, 1, 1, 9),
                datetime(2, 2, 2, 2, 2, 2, 9),
                datetime(3, 3, 3, 3, 3, 3, 9),
                None,
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_replace_int_date() -> None:
    df = pl.DataFrame(
        {
            "a": [
                date(1, 1, 1),
                date(2, 2, 2),
                date(3, 3, 3),
                None,
            ]
        }
    )
    result = df.select(
        pl.col("a").dt.replace().alias("no_change"),
        pl.col("a").dt.replace(year=9).alias("year"),
        pl.col("a").dt.replace(month=9).alias("month"),
        pl.col("a").dt.replace(day=9).alias("day"),
    )
    expected = pl.DataFrame(
        {
            "no_change": [
                date(1, 1, 1),
                date(2, 2, 2),
                date(3, 3, 3),
                None,
            ],
            "year": [
                date(9, 1, 1),
                date(9, 2, 2),
                date(9, 3, 3),
                None,
            ],
            "month": [
                date(1, 9, 1),
                date(2, 9, 2),
                date(3, 9, 3),
                None,
            ],
            "day": [
                date(1, 1, 9),
                date(2, 2, 9),
                date(3, 3, 9),
                None,
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_replace_ambiguous() -> None:
    # Value to be replaced by an ambiguous hour.
    value = pl.select(
        pl.datetime(2020, 10, 25, 5, time_zone="Europe/London")
    ).to_series()

    input = [2020, 10, 25, 1]
    tz = "Europe/London"

    # earliest
    expected = pl.select(
        pl.datetime(*input, time_zone=tz, ambiguous="earliest")
    ).to_series()
    result = value.dt.replace(hour=1, ambiguous="earliest")
    assert_series_equal(result, expected)

    # latest
    expected = pl.select(
        pl.datetime(*input, time_zone=tz, ambiguous="latest")
    ).to_series()
    result = value.dt.replace(hour=1, ambiguous="latest")
    assert_series_equal(result, expected)

    # null
    expected = pl.select(
        pl.datetime(*input, time_zone=tz, ambiguous="null")
    ).to_series()
    result = value.dt.replace(hour=1, ambiguous="null")
    assert_series_equal(result, expected)

    # raise
    with pytest.raises(
        ComputeError,
        match=(
            "datetime '2020-10-25 01:00:00' is ambiguous in time zone 'Europe/London'. "
            "Please use `ambiguous` to tell how it should be localized."
        ),
    ):
        value.dt.replace(hour=1, ambiguous="raise")


def test_replace_datetime_preserve_ns() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series(["2020-01-01T00:00:00.123456789"] * 2).cast(
                pl.Datetime("ns")
            ),
            "year": [2021, None],
            "microsecond": [50, None],
        }
    )

    result = df.select(
        year=pl.col("a").dt.replace(year="year"),
        us=pl.col("a").dt.replace(microsecond="microsecond"),
    )

    expected = pl.DataFrame(
        {
            "year": pl.Series(
                [
                    "2021-01-01T00:00:00.123456789",
                    "2020-01-01T00:00:00.123456789",
                ]
            ).cast(pl.Datetime("ns")),
            "us": pl.Series(
                [
                    "2020-01-01T00:00:00.000050",
                    "2020-01-01T00:00:00.123456789",
                ]
            ).cast(pl.Datetime("ns")),
        }
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
@pytest.mark.parametrize("tzinfo", [None, "Africa/Nairobi", "America/New_York"])
def test_replace_preserve_tu_and_tz(tu: TimeUnit, tzinfo: str) -> None:
    s = pl.Series(
        [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        dtype=pl.Datetime(time_unit=tu, time_zone=tzinfo),
    )
    result = s.dt.replace(year=2000)
    assert result.dtype.time_unit == tu  # type: ignore[attr-defined]
    assert result.dtype.time_zone == tzinfo  # type: ignore[attr-defined]
