from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.dependencies import _ZONEINFO_AVAILABLE
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
elif _ZONEINFO_AVAILABLE:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from polars.type_aliases import TimeUnit


@pytest.fixture()
def series_of_int_dates() -> pl.Series:
    return pl.Series([10000, 20000, 30000], dtype=pl.Date)


@pytest.fixture()
def series_of_str_dates() -> pl.Series:
    return pl.Series(["2020-01-01 00:00:00.000000000", "2020-02-02 03:20:10.987654321"])


def test_dt_to_string(series_of_int_dates: pl.Series) -> None:
    expected_str_dates = pl.Series(["1997-05-19", "2024-10-04", "2052-02-20"])

    assert series_of_int_dates.dtype == pl.Date
    assert_series_equal(series_of_int_dates.dt.to_string("%F"), expected_str_dates)

    # Check strftime alias as well
    assert_series_equal(series_of_int_dates.dt.strftime("%F"), expected_str_dates)


@pytest.mark.parametrize(
    ("unit_attr", "expected"),
    [
        ("year", pl.Series(values=[1997, 2024, 2052], dtype=pl.Int32)),
        ("iso_year", pl.Series(values=[1997, 2024, 2052], dtype=pl.Int32)),
        ("quarter", pl.Series(values=[2, 4, 1], dtype=pl.Int8)),
        ("month", pl.Series(values=[5, 10, 2], dtype=pl.Int8)),
        ("week", pl.Series(values=[21, 40, 8], dtype=pl.Int8)),
        ("day", pl.Series(values=[19, 4, 20], dtype=pl.Int8)),
        ("weekday", pl.Series(values=[1, 5, 2], dtype=pl.Int8)),
        ("ordinal_day", pl.Series(values=[139, 278, 51], dtype=pl.Int16)),
    ],
)
def test_dt_extract_datetime_component(
    unit_attr: str,
    expected: pl.Series,
    series_of_int_dates: pl.Series,
) -> None:
    assert_series_equal(getattr(series_of_int_dates.dt, unit_attr)(), expected)


@pytest.mark.parametrize(
    ("unit_attr", "expected"),
    [
        ("hour", pl.Series(values=[0, 3], dtype=pl.Int8)),
        ("minute", pl.Series(values=[0, 20], dtype=pl.Int8)),
        ("second", pl.Series(values=[0, 10], dtype=pl.Int8)),
        ("millisecond", pl.Series(values=[0, 987], dtype=pl.Int32)),
        ("microsecond", pl.Series(values=[0, 987654], dtype=pl.Int32)),
        ("nanosecond", pl.Series(values=[0, 987654321], dtype=pl.Int32)),
    ],
)
def test_strptime_extract_times(
    unit_attr: str,
    expected: pl.Series,
    series_of_int_dates: pl.Series,
    series_of_str_dates: pl.Series,
) -> None:
    s = series_of_str_dates.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S.%9f")

    assert_series_equal(getattr(s.dt, unit_attr)(), expected)


@pytest.mark.parametrize("time_zone", [None, "Asia/Kathmandu"])
@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("date", date(2022, 1, 1)),
        ("time", time(23)),
    ],
)
def test_dt_date_and_time(
    attribute: str, time_zone: None | str, expected: date | time
) -> None:
    ser = pl.Series([datetime(2022, 1, 1, 23)]).dt.replace_time_zone(time_zone)
    result = getattr(ser.dt, attribute)().item()
    assert result == expected


@pytest.mark.parametrize("time_zone", [None, "Asia/Kathmandu"])
@pytest.mark.parametrize("time_unit", ["us", "ns", "ms"])
def test_dt_datetime(time_zone: str | None, time_unit: TimeUnit) -> None:
    ser = (
        pl.Series([datetime(2022, 1, 1, 23)])
        .dt.cast_time_unit(time_unit)
        .dt.replace_time_zone(time_zone)
    )
    result = ser.dt.datetime()
    expected = datetime(2022, 1, 1, 23)
    assert result.dtype == pl.Datetime(time_unit, None)
    assert result.item() == expected


@pytest.mark.parametrize(
    ("time_zone", "expected"),
    [
        (None, True),
        ("Asia/Kathmandu", False),
        ("UTC", True),
    ],
)
@pytest.mark.parametrize("attribute", ["datetime", "date"])
def test_local_datetime_sortedness(
    time_zone: str | None, expected: bool, attribute: str
) -> None:
    ser = (pl.Series([datetime(2022, 1, 1, 23)]).dt.replace_time_zone(time_zone)).sort()
    result = getattr(ser.dt, attribute)()
    assert result.flags["SORTED_ASC"] == expected
    assert result.flags["SORTED_DESC"] is False


@pytest.mark.parametrize("time_zone", [None, "Asia/Kathmandu", "UTC"])
def test_local_time_sortedness(time_zone: str | None) -> None:
    ser = (pl.Series([datetime(2022, 1, 1, 23)]).dt.replace_time_zone(time_zone)).sort()
    result = ser.dt.time()
    assert result.flags["SORTED_ASC"] is False
    assert result.flags["SORTED_DESC"] is False


@pytest.mark.parametrize(
    ("time_zone", "offset", "expected"),
    [
        (None, "1d", True),
        ("Asia/Kathmandu", "1d", False),
        ("UTC", "1d", True),
        (None, "1mo", True),
        ("Asia/Kathmandu", "1mo", False),
        ("UTC", "1mo", True),
        (None, "1w", True),
        ("Asia/Kathmandu", "1w", False),
        ("UTC", "1w", True),
        (None, "1h", True),
        ("Asia/Kathmandu", "1h", True),
        ("UTC", "1h", True),
    ],
)
def test_offset_by_sortedness(
    time_zone: str | None, offset: str, expected: bool
) -> None:
    # create 2 values, as a single value is always sorted
    ser = (
        pl.Series(
            [datetime(2022, 1, 1, 22), datetime(2022, 1, 1, 22)]
        ).dt.replace_time_zone(time_zone)
    ).sort()
    result = ser.dt.offset_by(offset)
    assert result.flags["SORTED_ASC"] == expected
    assert result.flags["SORTED_DESC"] is False


def test_dt_datetime_date_time_invalid() -> None:
    with pytest.raises(ComputeError, match="expected Datetime"):
        pl.Series([date(2021, 1, 2)]).dt.datetime()
    with pytest.raises(ComputeError, match="expected Datetime or Date"):
        pl.Series([time(23)]).dt.date()
    with pytest.raises(ComputeError, match="expected Datetime"):
        pl.Series([time(23)]).dt.datetime()
    with pytest.raises(ComputeError, match="expected Datetime or Date"):
        pl.Series([timedelta(1)]).dt.date()
    with pytest.raises(ComputeError, match="expected Datetime"):
        pl.Series([timedelta(1)]).dt.datetime()
    with pytest.raises(ComputeError, match="expected Datetime, Date, or Time"):
        pl.Series([timedelta(1)]).dt.time()


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (datetime(2022, 3, 15, 3), datetime(2022, 3, 1, 3)),
        (datetime(2022, 3, 15, 3, 2, 1, 123000), datetime(2022, 3, 1, 3, 2, 1, 123000)),
        (datetime(2022, 3, 15), datetime(2022, 3, 1)),
        (datetime(2022, 3, 1), datetime(2022, 3, 1)),
    ],
)
@pytest.mark.parametrize(
    ("tzinfo", "time_zone"),
    [
        (None, None),
        (ZoneInfo("Asia/Kathmandu"), "Asia/Kathmandu"),
    ],
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_month_start_datetime(
    dt: datetime,
    expected: datetime,
    time_unit: TimeUnit,
    tzinfo: ZoneInfo | None,
    time_zone: str | None,
) -> None:
    ser = pl.Series([dt]).dt.replace_time_zone(time_zone).dt.cast_time_unit(time_unit)
    result = ser.dt.month_start().item()
    assert result == expected.replace(tzinfo=tzinfo)


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (date(2022, 3, 15), date(2022, 3, 1)),
        (date(2022, 3, 31), date(2022, 3, 1)),
    ],
)
def test_month_start_date(dt: date, expected: date) -> None:
    ser = pl.Series([dt])
    result = ser.dt.month_start().item()
    assert result == expected


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (datetime(2022, 3, 15, 3), datetime(2022, 3, 31, 3)),
        (
            datetime(2022, 3, 15, 3, 2, 1, 123000),
            datetime(2022, 3, 31, 3, 2, 1, 123000),
        ),
        (datetime(2022, 3, 15), datetime(2022, 3, 31)),
        (datetime(2022, 3, 31), datetime(2022, 3, 31)),
    ],
)
@pytest.mark.parametrize(
    ("tzinfo", "time_zone"),
    [
        (None, None),
        (ZoneInfo("Asia/Kathmandu"), "Asia/Kathmandu"),
    ],
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_month_end_datetime(
    dt: datetime,
    expected: datetime,
    time_unit: TimeUnit,
    tzinfo: ZoneInfo | None,
    time_zone: str | None,
) -> None:
    ser = pl.Series([dt]).dt.replace_time_zone(time_zone).dt.cast_time_unit(time_unit)
    result = ser.dt.month_end().item()
    assert result == expected.replace(tzinfo=tzinfo)


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        (date(2022, 3, 15), date(2022, 3, 31)),
        (date(2022, 3, 31), date(2022, 3, 31)),
    ],
)
def test_month_end_date(dt: date, expected: date) -> None:
    ser = pl.Series([dt])
    result = ser.dt.month_end().item()
    assert result == expected


def test_month_start_end_invalid() -> None:
    ser = pl.Series([time(1, 2, 3)])
    with pytest.raises(
        InvalidOperationError,
        match=r"`month_start` operation not supported for dtype `time` \(expected: date/datetime\)",
    ):
        ser.dt.month_start()
    with pytest.raises(
        InvalidOperationError,
        match=r"`month_end` operation not supported for dtype `time` \(expected: date/datetime\)",
    ):
        ser.dt.month_end()


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_base_utc_offset(time_unit: TimeUnit) -> None:
    ser = pl.datetime_range(
        datetime(2011, 12, 29),
        datetime(2012, 1, 1),
        "2d",
        time_zone="Pacific/Apia",
        eager=True,
    ).dt.cast_time_unit(time_unit)
    result = ser.dt.base_utc_offset().rename("base_utc_offset")
    expected = pl.Series(
        "base_utc_offset",
        [-11 * 3600 * 1000, 13 * 3600 * 1000],
        dtype=pl.Duration("ms"),
    )
    assert_series_equal(result, expected)


def test_base_utc_offset_lazy_schema() -> None:
    ser = pl.datetime_range(
        datetime(2020, 10, 25),
        datetime(2020, 10, 26),
        time_zone="Europe/London",
        eager=True,
    )
    df = pl.DataFrame({"ts": ser}).lazy()
    result = df.with_columns(base_utc_offset=pl.col("ts").dt.base_utc_offset()).schema
    expected = {
        "ts": pl.Datetime(time_unit="us", time_zone="Europe/London"),
        "base_utc_offset": pl.Duration(time_unit="ms"),
    }
    assert result == expected


def test_base_utc_offset_invalid() -> None:
    ser = pl.datetime_range(datetime(2020, 10, 25), datetime(2020, 10, 26), eager=True)
    with pytest.raises(
        InvalidOperationError,
        match=r"`base_utc_offset` operation not supported for dtype `datetime\[μs\]` \(expected: time-zone-aware datetime\)",
    ):
        ser.dt.base_utc_offset().rename("base_utc_offset")


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_dst_offset(time_unit: TimeUnit) -> None:
    ser = pl.datetime_range(
        datetime(2020, 10, 25),
        datetime(2020, 10, 26),
        time_zone="Europe/London",
        eager=True,
    ).dt.cast_time_unit(time_unit)
    result = ser.dt.dst_offset().rename("dst_offset")
    expected = pl.Series("dst_offset", [3_600 * 1_000, 0], dtype=pl.Duration("ms"))
    assert_series_equal(result, expected)


def test_dst_offset_lazy_schema() -> None:
    ser = pl.datetime_range(
        datetime(2020, 10, 25),
        datetime(2020, 10, 26),
        time_zone="Europe/London",
        eager=True,
    )
    df = pl.DataFrame({"ts": ser}).lazy()
    result = df.with_columns(dst_offset=pl.col("ts").dt.dst_offset()).schema
    expected = {
        "ts": pl.Datetime(time_unit="us", time_zone="Europe/London"),
        "dst_offset": pl.Duration(time_unit="ms"),
    }
    assert result == expected


def test_dst_offset_invalid() -> None:
    ser = pl.datetime_range(datetime(2020, 10, 25), datetime(2020, 10, 26), eager=True)
    with pytest.raises(
        InvalidOperationError,
        match=r"`dst_offset` operation not supported for dtype `datetime\[μs\]` \(expected: time-zone-aware datetime\)",
    ):
        ser.dt.dst_offset().rename("dst_offset")


@pytest.mark.parametrize(
    ("time_unit", "expected"),
    [
        ("d", pl.Series(values=[18262, 18294], dtype=pl.Int32)),
        ("s", pl.Series(values=[1_577_836_800, 1_580_613_610], dtype=pl.Int64)),
        (
            "ms",
            pl.Series(values=[1_577_836_800_000, 1_580_613_610_987], dtype=pl.Int64),
        ),
    ],
)
def test_strptime_epoch(
    time_unit: TimeUnit,
    expected: pl.Series,
    series_of_str_dates: pl.Series,
) -> None:
    s = series_of_str_dates.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S.%9f")

    assert_series_equal(s.dt.epoch(time_unit=time_unit), expected)


def test_strptime_fractional_seconds(series_of_str_dates: pl.Series) -> None:
    s = series_of_str_dates.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S.%9f")

    assert_series_equal(
        s.dt.second(fractional=True),
        pl.Series([0.0, 10.987654321], dtype=pl.Float64),
    )


@pytest.mark.parametrize(
    ("unit_attr", "expected"),
    [
        ("total_days", pl.Series([1])),
        ("total_hours", pl.Series([24])),
        ("total_minutes", pl.Series([24 * 60])),
        ("total_seconds", pl.Series([3600 * 24])),
        ("total_milliseconds", pl.Series([3600 * 24 * int(1e3)])),
        ("total_microseconds", pl.Series([3600 * 24 * int(1e6)])),
        ("total_nanoseconds", pl.Series([3600 * 24 * int(1e9)])),
    ],
)
def test_duration_extract_times(
    unit_attr: str,
    expected: pl.Series,
) -> None:
    duration = pl.Series([datetime(2022, 1, 2)]) - pl.Series([datetime(2022, 1, 1)])

    assert_series_equal(getattr(duration.dt, unit_attr)(), expected)


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
def test_duration_extract_times_deprecated_methods(
    unit_attr: str,
    expected: pl.Series,
) -> None:
    duration = pl.Series([datetime(2022, 1, 2)]) - pl.Series([datetime(2022, 1, 1)])

    with pytest.deprecated_call():
        assert_series_equal(getattr(duration.dt, unit_attr)(), expected)
    with pytest.deprecated_call():
        # Test Expr case too
        result_df = pl.select(getattr(pl.lit(duration).dt, unit_attr)())
        assert_series_equal(result_df[result_df.columns[0]], expected)


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
    time_unit: TimeUnit,
    every: str | timedelta,
) -> None:
    start, stop = datetime(2022, 1, 1), datetime(2022, 1, 2)
    s = pl.datetime_range(
        start,
        stop,
        timedelta(minutes=30),
        time_unit=time_unit,
        eager=True,
    ).alias(f"dates[{time_unit}]")

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
    time_unit: TimeUnit,
    every: str | timedelta,
) -> None:
    start, stop = datetime(2022, 1, 1), datetime(2022, 1, 2)
    s = pl.datetime_range(
        start,
        stop,
        timedelta(minutes=30),
        time_unit=time_unit,
        eager=True,
    ).alias(f"dates[{time_unit}]")

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
    time_unit: TimeUnit,
    date_in_that_unit: list[int],
) -> None:
    dates = pl.Series([datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])

    assert dates.dt.cast_time_unit(time_unit).cast(int).to_list() == date_in_that_unit


def test_epoch_matches_timestamp() -> None:
    dates = pl.Series([datetime(2001, 1, 1), datetime(2001, 2, 1, 10, 8, 9)])

    for unit in DTYPE_TEMPORAL_UNITS:
        assert_series_equal(dates.dt.epoch(unit), dates.dt.timestamp(unit))

    assert_series_equal(dates.dt.epoch("s"), dates.dt.timestamp("ms") // 1000)
    assert_series_equal(
        dates.dt.epoch("d"),
        (dates.dt.timestamp("ms") // (1000 * 3600 * 24)).cast(pl.Int32),
    )


@pytest.mark.parametrize(
    ("tzinfo", "time_zone"),
    [(None, None), (ZoneInfo("Asia/Kathmandu"), "Asia/Kathmandu")],
)
def test_date_time_combine(tzinfo: ZoneInfo | None, time_zone: str | None) -> None:
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
    df = df.with_columns(pl.col("dtm").dt.replace_time_zone(time_zone))

    # Combine datetime/date with time
    df = df.select(
        [
            pl.col("dtm").dt.combine(pl.col("tm")).alias("d1"),  # datetime & time
            pl.col("dt").dt.combine(pl.col("tm")).alias("d2"),  # date & time
            pl.col("dt").dt.combine(time(4, 5, 6)).alias("d3"),  # date & specified time
        ]
    )

    # Assert that the new columns have the expected values and datatypes
    expected_dict = {
        "d1": [  # Time component should be overwritten by `tm` values
            datetime(2022, 12, 31, 1, 2, 3, 456000, tzinfo=tzinfo),
            datetime(2023, 7, 5, 7, 8, 9, 101000, tzinfo=tzinfo),
        ],
        "d2": [  # Both date and time components combined "as-is" into new datetime
            datetime(2022, 10, 10, 1, 2, 3, 456000),
            datetime(2022, 7, 5, 7, 8, 9, 101000),
        ],
        "d3": [  # New datetime should use specified time component
            datetime(2022, 10, 10, 4, 5, 6),
            datetime(2022, 7, 5, 4, 5, 6),
        ],
    }
    assert df.to_dict(as_series=False) == expected_dict

    expected_schema = {
        "d1": pl.Datetime("us", time_zone),
        "d2": pl.Datetime("us"),
        "d3": pl.Datetime("us"),
    }
    assert df.schema == expected_schema


def test_combine_unsupported_types() -> None:
    with pytest.raises(ComputeError, match="expected Date or Datetime, got time"):
        pl.Series([time(1, 2)]).dt.combine(time(3, 4))


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("time_zone", ["Asia/Kathmandu", None])
def test_combine_lazy_schema_datetime(
    time_zone: str | None,
    time_unit: TimeUnit,
) -> None:
    df = pl.DataFrame({"ts": pl.Series([datetime(2020, 1, 1)])})
    df = df.with_columns(pl.col("ts").dt.replace_time_zone(time_zone))
    result = (
        df.lazy()
        .select(pl.col("ts").dt.combine(time(1, 2, 3), time_unit=time_unit))
        .dtypes
    )
    expected = [pl.Datetime(time_unit, time_zone)]
    assert result == expected


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_combine_lazy_schema_date(time_unit: TimeUnit) -> None:
    df = pl.DataFrame({"ts": pl.Series([date(2020, 1, 1)])})
    result = (
        df.lazy()
        .select(pl.col("ts").dt.combine(time(1, 2, 3), time_unit=time_unit))
        .dtypes
    )
    expected = [pl.Datetime(time_unit, None)]
    assert result == expected


def test_is_leap_year() -> None:
    assert pl.datetime_range(
        datetime(1990, 1, 1), datetime(2004, 1, 1), "1y", eager=True
    ).dt.is_leap_year().to_list() == [
        False,
        False,
        True,  # 1992
        False,
        False,
        False,
        True,  # 1996
        False,
        False,
        False,
        True,  # 2000
        False,
        False,
        False,
        True,  # 2004
    ]


def test_quarter() -> None:
    assert pl.datetime_range(
        datetime(2022, 1, 1), datetime(2022, 12, 1), "1mo", eager=True
    ).dt.quarter().to_list() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]


def test_date_offset() -> None:
    df = pl.DataFrame(
        {
            "dates": pl.datetime_range(
                datetime(2000, 1, 1), datetime(2020, 1, 1), "1y", eager=True
            )
        }
    )

    # Add two new columns to the DataFrame using the offset_by() method
    df = df.with_columns(
        [
            df["dates"].dt.offset_by("1y").alias("date_plus_1y"),
            df["dates"].dt.offset_by("-1y2mo").alias("date_min"),
        ]
    )

    # Assert that the day of the month for all the dates in new columns is 1
    assert (df["date_plus_1y"].dt.day() == 1).all()
    assert (df["date_min"].dt.day() == 1).all()

    # Assert that the 'date_min' column contains the expected list of dates
    expected_dates = [datetime(year, 11, 1, 0, 0) for year in range(1998, 2019)]
    assert df["date_min"].to_list() == expected_dates


@pytest.mark.parametrize("time_zone", ["US/Central", None])
def test_offset_by_crossing_dst(time_zone: str | None) -> None:
    ser = pl.Series([datetime(2021, 11, 7)]).dt.replace_time_zone(time_zone)
    result = ser.dt.offset_by("1d")
    expected = pl.Series([datetime(2021, 11, 8)]).dt.replace_time_zone(time_zone)
    assert_series_equal(result, expected)


def test_negative_offset_by_err_msg_8464() -> None:
    result = pl.Series([datetime(2022, 3, 30)]).dt.offset_by("-1mo")
    expected = pl.Series([datetime(2022, 2, 28)])
    assert_series_equal(result, expected)


def test_offset_by_truncate_sorted_flag() -> None:
    s = pl.Series([datetime(2001, 1, 1), datetime(2001, 1, 2)])
    s = s.set_sorted()

    assert s.flags["SORTED_ASC"]
    s1 = s.dt.offset_by("1d")
    assert s1.to_list() == [datetime(2001, 1, 2), datetime(2001, 1, 3)]
    assert s1.flags["SORTED_ASC"]
    s2 = s1.dt.truncate("1mo")
    assert s2.flags["SORTED_ASC"]


def test_offset_by_broadcasting() -> None:
    # test broadcast lhs
    df = pl.DataFrame(
        {
            "offset": ["1d", "10d", "3d", None],
        }
    )
    result = df.select(
        d1=pl.lit(datetime(2020, 10, 25)).dt.offset_by(pl.col("offset")),
        d2=pl.lit(datetime(2020, 10, 25))
        .dt.cast_time_unit("ms")
        .dt.offset_by(pl.col("offset")),
        d3=pl.lit(datetime(2020, 10, 25))
        .dt.replace_time_zone("Europe/London")
        .dt.offset_by(pl.col("offset")),
        d4=pl.lit(datetime(2020, 10, 25)).dt.date().dt.offset_by(pl.col("offset")),
        d5=pl.lit(None, dtype=pl.Datetime).dt.offset_by(pl.col("offset")),
    )
    expected_dict = {
        "d1": [
            datetime(2020, 10, 26),
            datetime(2020, 11, 4),
            datetime(2020, 10, 28),
            None,
        ],
        "d2": [
            datetime(2020, 10, 26),
            datetime(2020, 11, 4),
            datetime(2020, 10, 28),
            None,
        ],
        "d3": [
            datetime(2020, 10, 26, tzinfo=ZoneInfo(key="Europe/London")),
            datetime(2020, 11, 4, tzinfo=ZoneInfo(key="Europe/London")),
            datetime(2020, 10, 28, tzinfo=ZoneInfo(key="Europe/London")),
            None,
        ],
        "d4": [
            datetime(2020, 10, 26).date(),
            datetime(2020, 11, 4).date(),
            datetime(2020, 10, 28).date(),
            None,
        ],
        "d5": [None, None, None, None],
    }
    assert result.to_dict(as_series=False) == expected_dict

    # test broadcast rhs
    df = pl.DataFrame({"dt": [datetime(2020, 10, 25), datetime(2021, 1, 2), None]})
    result = df.select(
        d1=pl.col("dt").dt.offset_by(pl.lit("1mo3d")),
        d2=pl.col("dt").dt.cast_time_unit("ms").dt.offset_by(pl.lit("1y1mo")),
        d3=pl.col("dt")
        .dt.replace_time_zone("Europe/London")
        .dt.offset_by(pl.lit("3d")),
        d4=pl.col("dt").dt.date().dt.offset_by(pl.lit("1y1mo1d")),
    )
    expected_dict = {
        "d1": [datetime(2020, 11, 28), datetime(2021, 2, 5), None],
        "d2": [datetime(2021, 11, 25), datetime(2022, 2, 2), None],
        "d3": [
            datetime(2020, 10, 28, tzinfo=ZoneInfo(key="Europe/London")),
            datetime(2021, 1, 5, tzinfo=ZoneInfo(key="Europe/London")),
            None,
        ],
        "d4": [datetime(2021, 11, 26).date(), datetime(2022, 2, 3).date(), None],
    }
    assert result.to_dict(as_series=False) == expected_dict

    # test all literal
    result = df.select(d=pl.lit(datetime(2021, 11, 26)).dt.offset_by("1mo1d"))
    assert result.to_dict(as_series=False) == {"d": [datetime(2021, 12, 27)]}


def test_offset_by_expressions() -> None:
    df = pl.DataFrame(
        {
            "a": [
                datetime(2020, 10, 25),
                datetime(2021, 1, 2),
                None,
                datetime(2021, 1, 4),
                None,
            ],
            "b": ["1d", "10d", "3d", None, None],
        }
    )
    df = df.sort("a")
    result = df.select(
        c=pl.col("a").dt.offset_by(pl.col("b")),
        d=pl.col("a").dt.cast_time_unit("ms").dt.offset_by(pl.col("b")),
        e=pl.col("a").dt.replace_time_zone("Europe/London").dt.offset_by(pl.col("b")),
        f=pl.col("a").dt.date().dt.offset_by(pl.col("b")),
    )

    expected = pl.DataFrame(
        {
            "c": [None, None, datetime(2020, 10, 26), datetime(2021, 1, 12), None],
            "d": [None, None, datetime(2020, 10, 26), datetime(2021, 1, 12), None],
            "e": [None, None, datetime(2020, 10, 26), datetime(2021, 1, 12), None],
            "f": [None, None, date(2020, 10, 26), date(2021, 1, 12), None],
        },
        schema_overrides={
            "d": pl.Datetime("ms"),
            "e": pl.Datetime(time_zone="Europe/London"),
        },
    )
    assert_frame_equal(result, expected)
    assert result.flags == {
        "c": {"SORTED_ASC": False, "SORTED_DESC": False},
        "d": {"SORTED_ASC": False, "SORTED_DESC": False},
        "e": {"SORTED_ASC": False, "SORTED_DESC": False},
        "f": {"SORTED_ASC": False, "SORTED_DESC": False},
    }

    # Check single-row cases
    for i in range(len(df)):
        df_slice = df[i : i + 1]
        result = df_slice.select(
            c=pl.col("a").dt.offset_by(pl.col("b")),
            d=pl.col("a").dt.cast_time_unit("ms").dt.offset_by(pl.col("b")),
            e=pl.col("a")
            .dt.replace_time_zone("Europe/London")
            .dt.offset_by(pl.col("b")),
            f=pl.col("a").dt.date().dt.offset_by(pl.col("b")),
        )
        assert_frame_equal(result, expected[i : i + 1])
        if df_slice["b"].item() is None:
            # Offset is None, so result will be all-None, so sortedness isn't preserved.
            assert result.flags == {
                "c": {"SORTED_ASC": False, "SORTED_DESC": False},
                "d": {"SORTED_ASC": False, "SORTED_DESC": False},
                "e": {"SORTED_ASC": False, "SORTED_DESC": False},
                "f": {"SORTED_ASC": False, "SORTED_DESC": False},
            }
        else:
            # For tz-aware, sortedness is not preserved.
            assert result.flags == {
                "c": {"SORTED_ASC": True, "SORTED_DESC": False},
                "d": {"SORTED_ASC": True, "SORTED_DESC": False},
                "e": {"SORTED_ASC": False, "SORTED_DESC": False},
                "f": {"SORTED_ASC": True, "SORTED_DESC": False},
            }


@pytest.mark.parametrize(
    ("duration", "input_date", "expected"),
    [
        ("1mo", date(2018, 1, 31), date(2018, 2, 28)),
        ("1y", date(2024, 2, 29), date(2025, 2, 28)),
        ("1y1mo", date(2024, 1, 30), date(2025, 2, 28)),
    ],
)
def test_offset_by_saturating_8217_8474(
    duration: str, input_date: date, expected: date
) -> None:
    result = pl.Series([input_date]).dt.offset_by(duration).item()
    assert result == expected


def test_year_empty_df() -> None:
    df = pl.DataFrame(pl.Series(name="date", dtype=pl.Date))
    assert df.select(pl.col("date").dt.year()).dtypes == [pl.Int32]


@pytest.mark.parametrize(
    "time_unit",
    ["ms", "us", "ns"],
    ids=["milliseconds", "microseconds", "nanoseconds"],
)
def test_weekday(time_unit: TimeUnit) -> None:
    friday = pl.Series([datetime(2023, 2, 17)])

    assert friday.dt.cast_time_unit(time_unit).dt.weekday()[0] == 5
    assert friday.cast(pl.Date).dt.weekday()[0] == 5


@pytest.mark.parametrize(
    ("values", "expected_median"),
    [
        ([], None),
        ([None, None], None),
        ([date(2022, 1, 1)], datetime(2022, 1, 1)),
        ([date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)], datetime(2022, 1, 2)),
        ([date(2022, 1, 1), date(2022, 1, 2), date(2024, 5, 15)], datetime(2022, 1, 2)),
        (
            [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2024, 5, 15)],
            datetime(2022, 1, 2),
        ),
    ],
    ids=[
        "empty",
        "Nones",
        "single",
        "spread_even",
        "spread_skewed",
        "spread_skewed_dt",
    ],
)
def test_median(values: list[date | None], expected_median: date | None) -> None:
    s = pl.Series(values)
    assert s.dt.median() == expected_median

    if s.dtype == pl.Datetime:
        assert s.median() == expected_median


@pytest.mark.parametrize(
    ("values", "expected_mean"),
    [
        ([], None),
        ([None, None], None),
        ([date(2022, 1, 1)], datetime(2022, 1, 1)),
        ([date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)], datetime(2022, 1, 2)),
        (
            [date(2022, 1, 1), date(2022, 1, 2), date(2024, 5, 15)],
            datetime(2022, 10, 16, 16, 0, 0),
        ),
        (
            [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2024, 5, 15)],
            datetime(2022, 10, 16, 16, 0, 0),
        ),
    ],
    ids=[
        "empty",
        "Nones",
        "single",
        "spread_even",
        "spread_skewed",
        "spread_skewed_dt",
    ],
)
def test_mean(
    values: list[date | datetime | None], expected_mean: date | datetime | None
) -> None:
    s = pl.Series(values)
    assert s.dt.mean() == expected_mean

    if s.dtype == pl.Datetime:
        assert s.mean() == expected_mean


@pytest.mark.parametrize(
    ("values", "expected_mean"),
    [
        (
            [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2024, 5, 15)],
            datetime(2022, 10, 16, 16, 0, 0),
        ),
    ],
    ids=["spread_skewed_dt"],
)
def test_datetime_mean_with_tu(values: list[datetime], expected_mean: datetime) -> None:
    assert pl.Series(values, dtype=pl.Datetime("ms")).mean() == expected_mean
    assert pl.Series(values, dtype=pl.Datetime("ms")).dt.mean() == expected_mean
    assert pl.Series(values, dtype=pl.Datetime("us")).mean() == expected_mean
    assert pl.Series(values, dtype=pl.Datetime("us")).dt.mean() == expected_mean
    assert pl.Series(values, dtype=pl.Datetime("ns")).mean() == expected_mean
    assert pl.Series(values, dtype=pl.Datetime("ns")).dt.mean() == expected_mean


def test_agg_expr_mean() -> None:
    df = pl.DataFrame(
        {
            "date": pl.Series(
                [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 4)], dtype=pl.Date
            ),
            "datetime_ms": pl.Series(
                [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 4)],
                dtype=pl.Datetime("ms"),
            ),
            "datetime_us": pl.Series(
                [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 4)],
                dtype=pl.Datetime("us"),
            ),
            "datetime_ns": pl.Series(
                [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 4)],
                dtype=pl.Datetime("ns"),
            ),
            "duration_ms": pl.Series(
                [timedelta(days=1), timedelta(days=2), timedelta(days=4)],
                dtype=pl.Duration("ms"),
            ),
            "duration_us": pl.Series(
                [timedelta(days=1), timedelta(days=2), timedelta(days=4)],
                dtype=pl.Duration("us"),
            ),
            "duration_ns": pl.Series(
                [timedelta(days=1), timedelta(days=2), timedelta(days=4)],
                dtype=pl.Duration("ns"),
            ),
        }
    )

    expected = pl.DataFrame(
        {
            "date": pl.Series(
                [datetime(2023, 1, 2, 8, 0, 0)],
                dtype=pl.Datetime("ms"),
            ),
            "datetime_ms": pl.Series(
                [datetime(2023, 1, 2, 8, 0, 0)], dtype=pl.Datetime("ms")
            ),
            "datetime_us": pl.Series(
                [datetime(2023, 1, 2, 8, 0, 0)], dtype=pl.Datetime("us")
            ),
            "datetime_ns": pl.Series(
                [datetime(2023, 1, 2, 8, 0, 0)], dtype=pl.Datetime("ns")
            ),
            "duration_ms": pl.Series(
                [timedelta(days=2, hours=8)], dtype=pl.Duration("ms")
            ),
            "duration_us": pl.Series(
                [timedelta(days=2, hours=8)], dtype=pl.Duration("us")
            ),
            "duration_ns": pl.Series(
                [timedelta(days=2, hours=8)], dtype=pl.Duration("ns")
            ),
        }
    )

    assert_frame_equal(df.select(pl.all().mean()), expected)


def test_agg_expr_median() -> None:
    df = pl.DataFrame(
        {
            "date": pl.Series(
                [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 4)], dtype=pl.Date
            ),
            "datetime_ms": pl.Series(
                [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 4)],
                dtype=pl.Datetime("ms"),
            ),
            "datetime_us": pl.Series(
                [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 4)],
                dtype=pl.Datetime("us"),
            ),
            "datetime_ns": pl.Series(
                [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 4)],
                dtype=pl.Datetime("ns"),
            ),
            "duration_ms": pl.Series(
                [timedelta(days=1), timedelta(days=2), timedelta(days=4)],
                dtype=pl.Duration("ms"),
            ),
            "duration_us": pl.Series(
                [timedelta(days=1), timedelta(days=2), timedelta(days=4)],
                dtype=pl.Duration("us"),
            ),
            "duration_ns": pl.Series(
                [timedelta(days=1), timedelta(days=2), timedelta(days=4)],
                dtype=pl.Duration("ns"),
            ),
        }
    )

    expected = pl.DataFrame(
        {
            "date": pl.Series(
                [datetime(2023, 1, 2, 0, 0, 0)],
                dtype=pl.Datetime("ms"),
            ),
            "datetime_ms": pl.Series(
                [datetime(2023, 1, 2, 0, 0, 0)], dtype=pl.Datetime("ms")
            ),
            "datetime_us": pl.Series(
                [datetime(2023, 1, 2, 0, 0, 0)], dtype=pl.Datetime("us")
            ),
            "datetime_ns": pl.Series(
                [datetime(2023, 1, 2, 0, 0, 0)], dtype=pl.Datetime("ns")
            ),
            "duration_ms": pl.Series([timedelta(days=2)], dtype=pl.Duration("ms")),
            "duration_us": pl.Series([timedelta(days=2)], dtype=pl.Duration("us")),
            "duration_ns": pl.Series([timedelta(days=2)], dtype=pl.Duration("ns")),
        }
    )

    assert_frame_equal(df.select(pl.all().median()), expected)
