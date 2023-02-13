from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars.internals.type_aliases import TimeUnit


def test_dt_strftime() -> None:
    s = pl.Series("a", [10000, 20000, 30000], dtype=pl.Date)
    assert s.dtype == pl.Date
    expected = pl.Series("a", ["1997-05-19", "2024-10-04", "2052-02-20"])
    assert_series_equal(s.dt.strftime("%F"), expected)


def test_dt_year_month_week_day_ordinal_day() -> None:
    s = pl.Series("a", [10000, 20000, 30000], dtype=pl.Date)

    assert_series_equal(s.dt.year(), pl.Series("a", [1997, 2024, 2052], dtype=pl.Int32))

    assert_series_equal(s.dt.month(), pl.Series("a", [5, 10, 2], dtype=pl.UInt32))
    assert_series_equal(s.dt.weekday(), pl.Series("a", [1, 5, 2], dtype=pl.UInt32))
    assert_series_equal(s.dt.week(), pl.Series("a", [21, 40, 8], dtype=pl.UInt32))
    assert_series_equal(s.dt.day(), pl.Series("a", [19, 4, 20], dtype=pl.UInt32))
    assert_series_equal(
        s.dt.ordinal_day(), pl.Series("a", [139, 278, 51], dtype=pl.UInt32)
    )

    assert s.dt.median() == date(2024, 10, 4)
    assert s.dt.mean() == date(2024, 10, 4)


def test_dt_datetimes() -> None:
    s = pl.Series(["2020-01-01 00:00:00.000000000", "2020-02-02 03:20:10.987654321"])
    s = s.str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S.%9f")

    # hours, minutes, seconds, milliseconds, microseconds, and nanoseconds
    assert_series_equal(s.dt.hour(), pl.Series("", [0, 3], dtype=pl.UInt32))
    assert_series_equal(s.dt.minute(), pl.Series("", [0, 20], dtype=pl.UInt32))
    assert_series_equal(s.dt.second(), pl.Series("", [0, 10], dtype=pl.UInt32))
    assert_series_equal(s.dt.millisecond(), pl.Series("", [0, 987], dtype=pl.UInt32))
    assert_series_equal(s.dt.microsecond(), pl.Series("", [0, 987654], dtype=pl.UInt32))
    assert_series_equal(
        s.dt.nanosecond(), pl.Series("", [0, 987654321], dtype=pl.UInt32)
    )

    # epoch methods
    assert_series_equal(
        s.dt.epoch(tu="d"), pl.Series("", [18262, 18294], dtype=pl.Int32)
    )
    assert_series_equal(
        s.dt.epoch(tu="s"),
        pl.Series("", [1_577_836_800, 1_580_613_610], dtype=pl.Int64),
    )
    assert_series_equal(
        s.dt.epoch(tu="ms"),
        pl.Series("", [1_577_836_800_000, 1_580_613_610_000], dtype=pl.Int64),
    )
    # fractional seconds
    assert_series_equal(
        s.dt.second(fractional=True),
        pl.Series("", [0.0, 10.987654321], dtype=pl.Float64),
    )


def test_duration_extract_times() -> None:
    a = pl.Series("a", [datetime(2021, 1, 1)])
    b = pl.Series("b", [datetime(2021, 1, 2)])

    duration = b - a
    expected = pl.Series("b", [1])
    assert_series_equal(duration.dt.days(), expected)

    expected = pl.Series("b", [24])
    assert_series_equal(duration.dt.hours(), expected)

    expected = pl.Series("b", [3600 * 24])
    assert_series_equal(duration.dt.seconds(), expected)

    expected = pl.Series("b", [3600 * 24 * int(1e3)])
    assert_series_equal(duration.dt.milliseconds(), expected)

    expected = pl.Series("b", [3600 * 24 * int(1e6)])
    assert_series_equal(duration.dt.microseconds(), expected)

    expected = pl.Series("b", [3600 * 24 * int(1e9)])
    assert_series_equal(duration.dt.nanoseconds(), expected)


def test_truncate() -> None:
    start = datetime(2001, 1, 1)
    stop = datetime(2001, 1, 2)

    s1 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[ms]", time_unit="ms"
    )
    s2 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[us]", time_unit="us"
    )
    s3 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[ns]", time_unit="ns"
    )

    # can pass strings and timedeltas
    for out in [
        s1.dt.truncate("1h"),
        s2.dt.truncate("1h0m0s"),
        s3.dt.truncate(timedelta(hours=1)),
    ]:
        assert out.dt[0] == start
        assert out.dt[1] == start
        assert out.dt[2] == start + timedelta(hours=1)
        assert out.dt[3] == start + timedelta(hours=1)
        # ...
        assert out.dt[-3] == stop - timedelta(hours=1)
        assert out.dt[-2] == stop - timedelta(hours=1)
        assert out.dt[-1] == stop


def test_round() -> None:
    start = datetime(2001, 1, 1)
    stop = datetime(2001, 1, 2)

    s1 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[ms]", time_unit="ms"
    )
    s2 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[us]", time_unit="us"
    )
    s3 = pl.date_range(
        start, stop, timedelta(minutes=30), name="dates[ns]", time_unit="ns"
    )

    # can pass strings and timedeltas
    for out in [
        s1.dt.round("1h"),
        s2.dt.round("1h0m0s"),
        s3.dt.round(timedelta(hours=1)),
    ]:
        assert out.dt[0] == start
        assert out.dt[1] == start + timedelta(hours=1)
        assert out.dt[2] == start + timedelta(hours=1)
        assert out.dt[3] == start + timedelta(hours=2)
        # ...
        assert out.dt[-3] == stop - timedelta(hours=1)
        assert out.dt[-2] == stop
        assert out.dt[-1] == stop


def test_cast_time_units() -> None:
    dates = pl.Series("dates", [datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])
    dates_in_ns = np.array([978307200000000000, 981022089000000000])

    assert dates.dt.cast_time_unit("ns").cast(int).to_list() == list(dates_in_ns)
    assert dates.dt.cast_time_unit("us").cast(int).to_list() == list(
        dates_in_ns // 1_000
    )
    assert dates.dt.cast_time_unit("ms").cast(int).to_list() == list(
        dates_in_ns // 1_000_000
    )


def test_epoch() -> None:
    dates = pl.Series("dates", [datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])

    for unit in DTYPE_TEMPORAL_UNITS:
        assert_series_equal(dates.dt.epoch(unit), dates.dt.timestamp(unit))

    assert_series_equal(dates.dt.epoch("s"), dates.dt.timestamp("ms") // 1000)
    assert_series_equal(
        dates.dt.epoch("d"),
        (dates.dt.timestamp("ms") // (1000 * 3600 * 24)).cast(pl.Int32),
    )


def test_date_time_combine() -> None:
    # test combining datetime/date and time (as expr/col and as literal)
    df = pl.DataFrame(
        {
            "dtm": [
                datetime(2022, 12, 31, 10, 30, 45),
                datetime(2023, 7, 5, 23, 59, 59),
            ],
            "dt": [date(2022, 10, 10), date(2022, 7, 5)],
            "tm": [time(1, 2, 3, 456000), time(7, 8, 9, 101000)],
        }
    ).select(
        [
            pl.col("dtm").dt.combine(pl.col("tm")).alias("d1"),
            pl.col("dt").dt.combine(pl.col("tm")).alias("d2"),
            pl.col("dt").dt.combine(time(4, 5, 6)).alias("d3"),
        ]
    )
    # if combining with datetime, the time component should be overwritten.
    # if combining with date, should write both parts 'as-is' into the new datetime.
    assert df.to_dict(False) == {
        "d1": [
            datetime(2022, 12, 31, 1, 2, 3, 456000),
            datetime(2023, 7, 5, 7, 8, 9, 101000),
        ],
        "d2": [
            datetime(2022, 10, 10, 1, 2, 3, 456000),
            datetime(2022, 7, 5, 7, 8, 9, 101000),
        ],
        "d3": [datetime(2022, 10, 10, 4, 5, 6), datetime(2022, 7, 5, 4, 5, 6)],
    }
    assert df.schema == {
        "d1": pl.Datetime("us"),
        "d2": pl.Datetime("us"),
        "d3": pl.Datetime("us"),
    }


def test_quarter() -> None:
    assert pl.date_range(
        datetime(2022, 1, 1), datetime(2022, 12, 1), "1mo"
    ).dt.quarter().to_list() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]


def test_date_offset() -> None:
    out = pl.DataFrame(
        {"dates": pl.date_range(datetime(2000, 1, 1), datetime(2020, 1, 1), "1y")}
    ).with_columns(
        [
            pl.col("dates").dt.offset_by("1y").alias("date_plus_1y"),
            pl.col("dates").dt.offset_by("-1y2mo").alias("date_min"),
        ]
    )

    assert (out["date_plus_1y"].dt.day() == 1).all()
    assert (out["date_min"].dt.day() == 1).all()
    assert out["date_min"].to_list() == [
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


def test_year_empty_df() -> None:
    df = pl.DataFrame(pl.Series(name="date", dtype=pl.Date))
    assert df.select(pl.col("date").dt.year()).dtypes == [pl.Int32]


def test_weekday() -> None:
    # monday
    s = pl.Series([datetime(2020, 1, 6)])

    time_units: list[TimeUnit] = ["ns", "us", "ms"]
    for tu in time_units:
        assert s.dt.cast_time_unit(tu).dt.weekday()[0] == 1

    assert s.cast(pl.Date).dt.weekday()[0] == 1


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([datetime(1969, 12, 31), datetime(1970, 1, 2)], datetime(1970, 1, 1)),
        ([None, None], None),
    ],
    ids=["datetime_dates", "Nones"],
)
def test_median(values: list[datetime | None], expected: datetime | None) -> None:
    result = pl.Series(values).cast(pl.Datetime).dt.median()
    assert result == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([datetime(1969, 12, 31), datetime(1970, 1, 2)], datetime(1970, 1, 1)),
        ([None, None], None),
    ],
    ids=["datetime_dates", "Nones"],
)
def test_mean(values: list[datetime | None], expected: datetime | None) -> None:
    result = pl.Series(values).cast(pl.Datetime).dt.median()
    assert result == expected
