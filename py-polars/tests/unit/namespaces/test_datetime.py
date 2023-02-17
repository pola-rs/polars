from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars.internals.type_aliases import TimeUnit


@pytest.fixture()
def date_2022_01_01() -> date:
    return date(2022, 1, 1)


@pytest.fixture()
def date_2022_01_02() -> date:
    return date(2022, 1, 2)


@pytest.fixture()
def series_of_dates() -> pl.Series:
    return pl.Series([10000, 20000, 30000], dtype=pl.Date)


def test_dt_strftime(series_of_dates: pl.Series) -> None:
    expected = pl.Series(["1997-05-19", "2024-10-04", "2052-02-20"])

    assert series_of_dates.dtype == pl.Date
    assert_series_equal(series_of_dates.dt.strftime("%F"), expected)


def test_dt_year_month_week_day_ordinal_day(
    series_of_dates: pl.Series,
) -> None:
    assert_series_equal(
        series_of_dates.dt.year(), pl.Series([1997, 2024, 2052], dtype=pl.Int32)
    )
    assert_series_equal(
        series_of_dates.dt.month(), pl.Series([5, 10, 2], dtype=pl.UInt32)
    )
    assert_series_equal(
        series_of_dates.dt.weekday(), pl.Series([1, 5, 2], dtype=pl.UInt32)
    )
    assert_series_equal(
        series_of_dates.dt.week(), pl.Series([21, 40, 8], dtype=pl.UInt32)
    )
    assert_series_equal(
        series_of_dates.dt.day(), pl.Series([19, 4, 20], dtype=pl.UInt32)
    )
    assert_series_equal(
        series_of_dates.dt.ordinal_day(), pl.Series([139, 278, 51], dtype=pl.UInt32)
    )


def test_dt_datetimes() -> None:
    s = pl.Series(["2020-01-01 00:00:00.000000000", "2020-02-02 03:20:10.987654321"])
    s = s.str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S.%9f")

    # hours, minutes, seconds, milliseconds, microseconds, and nanoseconds
    assert_series_equal(s.dt.hour(), pl.Series([0, 3], dtype=pl.UInt32))
    assert_series_equal(s.dt.minute(), pl.Series([0, 20], dtype=pl.UInt32))
    assert_series_equal(s.dt.second(), pl.Series([0, 10], dtype=pl.UInt32))
    assert_series_equal(s.dt.millisecond(), pl.Series([0, 987], dtype=pl.UInt32))
    assert_series_equal(s.dt.microsecond(), pl.Series([0, 987654], dtype=pl.UInt32))
    assert_series_equal(s.dt.nanosecond(), pl.Series([0, 987654321], dtype=pl.UInt32))

    # epoch methods
    assert_series_equal(s.dt.epoch(tu="d"), pl.Series([18262, 18294], dtype=pl.Int32))
    assert_series_equal(
        s.dt.epoch(tu="s"),
        pl.Series([1_577_836_800, 1_580_613_610], dtype=pl.Int64),
    )
    assert_series_equal(
        s.dt.epoch(tu="ms"),
        pl.Series([1_577_836_800_000, 1_580_613_610_000], dtype=pl.Int64),
    )
    # fractional seconds
    assert_series_equal(
        s.dt.second(fractional=True),
        pl.Series([0.0, 10.987654321], dtype=pl.Float64),
    )


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("days", [1]),
        ("hours", [24]),
        ("seconds", [3600 * 24]),
        ("milliseconds", [3600 * 24 * int(1e3)]),
        ("microseconds", [3600 * 24 * int(1e6)]),
        ("nanoseconds", [3600 * 24 * int(1e9)]),
    ],
)
def test_duration_extract_times(
    unit: str, expected: list[int], date_2022_01_01: date, date_2022_01_02: date
) -> None:
    duration = pl.Series([date_2022_01_02]) - pl.Series([date_2022_01_01])

    assert_series_equal(getattr(duration.dt, unit)(), pl.Series(expected))


@pytest.mark.parametrize(
    ("time_unit", "every"),
    [
        ("ms", "1h"),
        ("us", "1h0m0s"),
        ("ns", timedelta(hours=1)),
    ],
    ids=["milliseconds", "microseconds", "nanoseconds"],
)
def test_truncate(
    time_unit: Literal["ms", "us", "ns"],
    every: str | timedelta,
    start: date = date_2022_01_01,
    stop: date = date_2022_01_02,
) -> None:
    s = pl.date_range(
        start,
        stop,
        timedelta(minutes=30),
        name=f"dates[{time_unit}]",
        time_unit=time_unit,
    )

    # can pass strings and time-deltas
    out = s.dt.truncate(every)
    assert out.dt[0] == start
    assert out.dt[1] == start
    assert out.dt[2] == start + timedelta(hours=1)
    assert out.dt[3] == start + timedelta(hours=1)
    # ...
    assert out.dt[-3] == stop - timedelta(hours=1)
    assert out.dt[-2] == stop - timedelta(hours=1)
    assert out.dt[-1] == stop


@pytest.mark.parametrize(
    ("time_unit", "every"),
    [
        ("ms", "1h"),
        ("us", "1h0m0s"),
        ("ns", timedelta(hours=1)),
    ],
    ids=["milliseconds", "microseconds", "nanoseconds"],
)
def test_round(
    time_unit: Literal["ms", "us", "ns"],
    every: str | timedelta,
    start: date = date_2022_01_01,
    stop: date = date_2022_01_02,
) -> None:
    s = pl.date_range(
        start,
        stop,
        timedelta(minutes=30),
        name=f"dates[{time_unit}]",
        time_unit=time_unit,
    )

    # can pass strings and time-deltas
    out = s.dt.round(every)
    assert out.dt[0] == start
    assert out.dt[1] == start + timedelta(hours=1)
    assert out.dt[2] == start + timedelta(hours=1)
    assert out.dt[3] == start + timedelta(hours=2)
    # ...
    assert out.dt[-3] == stop - timedelta(hours=1)
    assert out.dt[-2] == stop
    assert out.dt[-1] == stop


def test_cast_time_units() -> None:
    dates = pl.Series([datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])
    dates_in_ns = np.array([978307200000000000, 981022089000000000])

    assert dates.dt.cast_time_unit("ns").cast(int).to_list() == list(dates_in_ns)
    assert dates.dt.cast_time_unit("us").cast(int).to_list() == list(
        dates_in_ns // 1_000
    )
    assert dates.dt.cast_time_unit("ms").cast(int).to_list() == list(
        dates_in_ns // 1_000_000
    )


def test_epoch() -> None:
    dates = pl.Series([datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])

    for unit in DTYPE_TEMPORAL_UNITS:
        assert_series_equal(dates.dt.epoch(unit), dates.dt.timestamp(unit))

    assert_series_equal(dates.dt.epoch("s"), dates.dt.timestamp("ms") // 1000)
    assert_series_equal(
        dates.dt.epoch("d"),
        (dates.dt.timestamp("ms") // (1000 * 3600 * 24)).cast(pl.Int32),
    )


def test_date_time_combine() -> None:
    # Define a DataFrame with columns for datetime, date, and time
    df = pl.DataFrame({
        "dtm": [
            datetime(2022, 12, 31, 10, 30, 45),
            datetime(2023, 7, 5, 23, 59, 59),
        ],
        "dt": [
            date(2022, 10, 10),
            date(2022, 7, 5),
        ],
        "tm": [
            time(1, 2, 3, 456000),
            time(7, 8, 9, 101000),
        ],
    })

    # Use the .select() method to combine datetime/date and time
    df = df.select([
        pl.col("dtm").dt.combine(pl.col("tm")).alias("d1"),  # Combine datetime and time
        pl.col("dt").dt.combine(pl.col("tm")).alias("d2"),  # Combine date and time
        pl.col("dt").dt.combine(time(4, 5, 6)).alias("d3"),  # Combine date and a specified time
    ])

    # Check that the new columns have the expected values and datatypes
    expected_dict = {
        "d1": [
            datetime(2022, 12, 31, 1, 2, 3, 456000),  # Time component should be overwritten by `tm`
            datetime(2023, 7, 5, 7, 8, 9, 101000),
        ],
        "d2": [
            datetime(2022, 10, 10, 1, 2, 3, 456000),
            datetime(2022, 7, 5, 7, 8, 9, 101000),
        ],
        "d3": [
            datetime(2022, 10, 10, 4, 5, 6),  # New datetime should use specified time component
            datetime(2022, 7, 5, 4, 5, 6),
        ],
    }
    assert df.to_dict(False) == expected_dict

    expected_schema = {
        "d1": pl.Datetime("us"),
        "d2": pl.Datetime("us"),
        "d3": pl.Datetime("us"),
    }
    assert df.schema == expected_schema


def test_quarter() -> None:
    assert pl.date_range(
        datetime(2022, 1, 1), datetime(2022, 12, 1), "1mo"
    ).dt.quarter().to_list() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]


def test_date_offset() -> None:
    df = pl.DataFrame({"dates": pl.date_range(datetime(2000, 1, 1), datetime(2020, 1, 1), "1y")})

    # Add two new columns to the DataFrame using the offset_by() method
    df = df.with_columns([
        df["dates"].dt.offset_by("1y").alias("date_plus_1y"),
        df["dates"].dt.offset_by("-1y2mo").alias("date_min"),
    ])

    # Assert that the day of the month for all dates in the 'date_plus_1y' column is 1
    assert (df["date_plus_1y"].dt.day() == 1).all()

    # Assert that the day of the month for all dates in the 'date_min' column is 1
    assert (df["date_min"].dt.day() == 1).all()

    # Assert that the 'date_min' column contains the expected list of dates
    expected_dates = [
        datetime(1998, 11, 1, 0, 0),
        datetime(1999, 11, 1, 0, 0),
        datetime(2000, 11, 1, 0, 0),
        datetime(2001, 11, 1, 0, 0),
        datetime(2002, 11, 1, 0, 0),
        datetime(2003, 11, 1, 0, 0),
        datetime(2004, 11, 1, 0, 0),
        datetime(2005, 11, 1, 0, 0),
        datetime(2006, 11, 1, 0, 0),
        datetime(2007, 11, 1, 0, 0),
        datetime(2008, 11, 1, 0, 0),
        datetime(2009, 11, 1, 0, 0),
        datetime(2010, 11, 1, 0, 0),
        datetime(2011, 11, 1, 0, 0),
        datetime(2012, 11, 1, 0, 0),
        datetime(2013, 11, 1, 0, 0),
        datetime(2014, 11, 1, 0, 0),
        datetime(2015, 11, 1, 0, 0),
        datetime(2016, 11, 1, 0, 0),
        datetime(2017, 11, 1, 0, 0),
        datetime(2018, 11, 1, 0, 0),
    ]
    assert df["date_min"].to_list() == expected_dates


def test_year_empty_df() -> None:
    df = pl.DataFrame(pl.Series(name="date", dtype=pl.Date))
    assert df.select(pl.col("date").dt.year()).dtypes == [pl.Int32]


@pytest.mark.parametrize(
    "time_unit",
    ["ms", "us", "ns"],
    ids=["milliseconds", "microseconds", "nanoseconds"],
)
def test_weekday(time_unit: Literal["ms", "us", "ns"], date_2022_01_01: date) -> None:
    friday = pl.Series([date_2022_01_01])

    assert friday.dt.cast_time_unit(time_unit).dt.weekday()[0] == 5
    assert friday.cast(pl.Date).dt.weekday()[0] == 5


@pytest.mark.parametrize(
    ("values", "expected_median"),
    [
        ([], None),
        ([None, None], None),
        ([date(2022, 1, 1)], date(2022, 1, 1)),
        ([date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)], date(2022, 1, 2)),
        ([date(2022, 1, 1), date(2022, 1, 2), date(2024, 5, 15)], date(2022, 1, 2)),
    ],
    ids=["empty", "Nones", "single", "spread_even", "spread_skewed"],
)
def test_median(values: list[date | None], expected_median: date | None) -> None:
    result = pl.Series(values).cast(pl.Date).dt.median()
    assert result == expected_median


@pytest.mark.parametrize(
    ("values", "expected_mean"),
    [
        ([], None),
        ([None, None], None),
        ([date(2022, 1, 1)], date(2022, 1, 1)),
        ([date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)], date(2022, 1, 2)),
        ([date(2022, 1, 1), date(2022, 1, 2), date(2024, 5, 15)], date(2022, 10, 16)),
    ],
    ids=["empty", "Nones", "single", "spread_even", "spread_skewed"],
)
def test_mean(values: list[date | None], expected_mean: date | None) -> None:
    result = pl.Series(values).cast(pl.Date).dt.mean()
    assert result == expected_mean
