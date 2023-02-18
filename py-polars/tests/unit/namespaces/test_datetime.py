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
def date_2022_01_01() -> datetime:
    return datetime(2022, 1, 1)


@pytest.fixture()
def date_2022_01_02() -> datetime:
    return datetime(2022, 1, 2)


@pytest.fixture()
def series_of_int_dates() -> pl.Series:
    return pl.Series([10000, 20000, 30000], dtype=pl.Date)


@pytest.fixture()
def series_of_str_dates() -> pl.Series:
    return pl.Series(["2020-01-01 00:00:00.000000000", "2020-02-02 03:20:10.987654321"])


def test_dt_strftime(series_of_int_dates: pl.Series) -> None:
    expected_str_dates = pl.Series(["1997-05-19", "2024-10-04", "2052-02-20"])

    assert series_of_int_dates.dtype == pl.Date
    assert_series_equal(series_of_int_dates.dt.strftime("%F"), expected_str_dates)


@pytest.mark.parametrize(
    ("unit_attr", "expected"),
    [
        ("year", pl.Series(values=[1997, 2024, 2052], dtype=pl.Int32)),
        ("month", pl.Series(values=[5, 10, 2], dtype=pl.UInt32)),
        ("week", pl.Series(values=[21, 40, 8], dtype=pl.UInt32)),
        ("day", pl.Series(values=[19, 4, 20], dtype=pl.UInt32)),
        ("ordinal_day", pl.Series(values=[139, 278, 51], dtype=pl.UInt32)),
    ],
)
def test_dt_extract_year_month_week_day_ordinal_day(
    unit_attr: str,
    expected: pl.Series,
    series_of_int_dates: pl.Series,
) -> None:
    assert_series_equal(getattr(series_of_int_dates.dt, unit_attr)(), expected)


@pytest.mark.parametrize(
    ("unit_attr", "expected"),
    [
        ("hour", pl.Series(values=[0, 3], dtype=pl.UInt32)),
        ("minute", pl.Series(values=[0, 20], dtype=pl.UInt32)),
        ("second", pl.Series(values=[0, 10], dtype=pl.UInt32)),
        ("millisecond", pl.Series(values=[0, 987], dtype=pl.UInt32)),
        ("microsecond", pl.Series(values=[0, 987654], dtype=pl.UInt32)),
        ("nanosecond", pl.Series(values=[0, 987654321], dtype=pl.UInt32)),
    ],
)
def test_strptime_extract_times(
    unit_attr: str,
    expected: pl.Series,
    series_of_int_dates: pl.Series,
    series_of_str_dates: pl.Series,
) -> None:
    s = series_of_str_dates.str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S.%9f")

    assert_series_equal(getattr(s.dt, unit_attr)(), expected)


@pytest.mark.parametrize(
    ("temporal_unit", "expected"),
    [
        ("d", pl.Series(values=[18262, 18294], dtype=pl.Int32)),
        ("s", pl.Series(values=[1_577_836_800, 1_580_613_610], dtype=pl.Int64)),
        (
            "ms",
            pl.Series(values=[1_577_836_800_000, 1_580_613_610_000], dtype=pl.Int64),
        ),
    ],
)
def test_strptime_epoch(
    temporal_unit: Literal["d", "s", "ms"],
    expected: pl.Series,
    series_of_str_dates: pl.Series,
) -> None:
    s = series_of_str_dates.str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S.%9f")

    assert_series_equal(s.dt.epoch(tu=temporal_unit), expected)


def test_strptime_fractional_seconds(series_of_str_dates: pl.Series):
    s = series_of_str_dates.str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S.%9f")

    assert_series_equal(
        s.dt.second(fractional=True),
        pl.Series([0.0, 10.987654321], dtype=pl.Float64),
    )


@pytest.mark.parametrize(
    ("unit_attr", "expected"),
    [
        ("days", pl.Series([1])),
        ("hours", pl.Series([24])),
        ("minutes", pl.Series([24 * 60])),
        ("seconds", pl.Series([3600 * 24])),
        ("milliseconds", pl.Series([3600 * 24 * int(1e3)])),
        ("microseconds", pl.Series([3600 * 24 * int(1e6)])),
        ("nanoseconds", pl.Series([3600 * 24 * int(1e9)])),
    ],
)
def test_duration_extract_times(
    unit_attr: str,
    expected: pl.Series,
    date_2022_01_01: datetime,
    date_2022_01_02: datetime,
) -> None:
    duration = pl.Series([date_2022_01_02]) - pl.Series([date_2022_01_01])

    assert_series_equal(getattr(duration.dt, unit_attr)(), expected)


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
    date_2022_01_01: datetime,
    date_2022_01_02: datetime,
) -> None:
    start, stop = date_2022_01_01, date_2022_01_02
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
    date_2022_01_01: datetime,
    date_2022_01_02: datetime,
) -> None:
    start, stop = date_2022_01_01, date_2022_01_02
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


@pytest.mark.parametrize(
    ("time_unit", "date_in_that_unit"),
    [
        ("ns", [978307200000000000, 981022089000000000]),
        ("us", [978307200000000, 981022089000000]),
        ("ms", [978307200000, 981022089000]),
    ],
    ids=["nanoseconds", "microseconds", "milliseconds"],
)
def test_cast_time_units(
    time_unit: Literal["ms", "us", "ns"],
    date_in_that_unit: list[int],
) -> None:
    dates = pl.Series([datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])

    assert dates.dt.cast_time_unit(time_unit).cast(int).to_list() == date_in_that_unit


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
    df = pl.DataFrame(
        {
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
        }
    )

    # Use the .select() method to combine datetime/date and time
    df = df.select(
        [
            pl.col("dtm")
            .dt.combine(pl.col("tm"))
            .alias("d1"),  # Combine datetime and time
            pl.col("dt").dt.combine(pl.col("tm")).alias("d2"),  # Combine date and time
            pl.col("dt")
            .dt.combine(time(4, 5, 6))
            .alias("d3"),  # Combine date and a specified time
        ]
    )

    # Check that the new columns have the expected values and datatypes
    expected_dict = {
        "d1": [
            datetime(
                2022, 12, 31, 1, 2, 3, 456000
            ),  # Time component should be overwritten by `tm`
            datetime(2023, 7, 5, 7, 8, 9, 101000),
        ],
        "d2": [
            datetime(2022, 10, 10, 1, 2, 3, 456000),
            datetime(2022, 7, 5, 7, 8, 9, 101000),
        ],
        "d3": [
            datetime(
                2022, 10, 10, 4, 5, 6
            ),  # New datetime should use specified time component
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
    df = pl.DataFrame(
        {"dates": pl.date_range(datetime(2000, 1, 1), datetime(2020, 1, 1), "1y")}
    )

    # Add two new columns to the DataFrame using the offset_by() method
    df = df.with_columns(
        [
            df["dates"].dt.offset_by("1y").alias("date_plus_1y"),
            df["dates"].dt.offset_by("-1y2mo").alias("date_min"),
        ]
    )

    # Assert that the day of the month for all dates in the 'date_plus_1y' column is 1
    assert (df["date_plus_1y"].dt.day() == 1).all()

    # Assert that the day of the month for all dates in the 'date_min' column is 1
    assert (df["date_min"].dt.day() == 1).all()

    # Assert that the 'date_min' column contains the expected list of dates
    expected_dates = [
        datetime(year, 11, 1, 0, 0) for year in range(1998, 2019)
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
def test_weekday(time_unit: Literal["ms", "us", "ns"]) -> None:
    friday = pl.Series([datetime(2023, 2, 17)])

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
