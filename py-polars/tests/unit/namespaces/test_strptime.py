from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone

import pytest

import polars as pl
from polars.datatypes import PolarsTemporalType
from polars.exceptions import ComputeError
from polars.internals.type_aliases import TimeUnit
from polars.testing import assert_series_equal

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
else:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo


def test_str_strptime() -> None:
    s = pl.Series(["2020-01-01", "2020-02-02"])
    expected = pl.Series([date(2020, 1, 1), date(2020, 2, 2)])
    assert_series_equal(s.str.strptime(pl.Date, "%Y-%m-%d"), expected)

    s = pl.Series(["2020-01-01 00:00:00", "2020-02-02 03:20:10"])
    expected = pl.Series(
        [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 2, 2, 3, 20, 10)]
    )
    assert_series_equal(s.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"), expected)

    s = pl.Series(["00:00:00", "03:20:10"])
    expected = pl.Series([0, 12010000000000], dtype=pl.Time)
    assert_series_equal(s.str.strptime(pl.Time, "%H:%M:%S"), expected)


def test_date_parse_omit_day() -> None:
    df = pl.DataFrame({"month": ["2022-01"]})
    assert df.select(pl.col("month").str.strptime(pl.Date, fmt="%Y-%m")).item() == date(
        2022, 1, 1
    )
    assert df.select(
        pl.col("month").str.strptime(pl.Datetime, fmt="%Y-%m")
    ).item() == datetime(2022, 1, 1)


def test_strptime_precision() -> None:
    s = pl.Series(
        "date", ["2022-09-12 21:54:36.789321456", "2022-09-13 12:34:56.987456321"]
    )
    ds = s.str.strptime(pl.Datetime)
    assert ds.cast(pl.Date) != None  # noqa: E711  (note: *deliberately* testing "!=")
    assert getattr(ds.dtype, "tu", None) == "us"

    time_units: list[TimeUnit] = ["ms", "us", "ns"]
    suffixes = ["%.3f", "%.6f", "%.9f"]
    test_data = zip(
        time_units,
        suffixes,
        (
            [789000000, 987000000],
            [789321000, 987456000],
            [789321456, 987456321],
        ),
    )
    for precision, suffix, expected_values in test_data:
        ds = s.str.strptime(pl.Datetime(precision), f"%Y-%m-%d %H:%M:%S{suffix}")
        assert getattr(ds.dtype, "tu", None) == precision
        assert ds.dt.nanosecond().to_list() == expected_values


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("ms", "123000000"), ("us", "123456000"), ("ns", "123456789")],
)
@pytest.mark.parametrize("fmt", ["%Y-%m-%d %H:%M:%S.%f", None])
def test_strptime_precision_with_time_unit(
    unit: TimeUnit, expected: str, fmt: str
) -> None:
    ser = pl.Series(["2020-01-01 00:00:00.123456789"])
    result = ser.str.strptime(pl.Datetime(unit), fmt=fmt).dt.strftime("%f")[0]
    assert result == expected


@pytest.mark.parametrize("fmt", ["%Y-%m-%dT%H:%M:%S", None])
def test_utc_with_tz_naive(fmt: str | None) -> None:
    with pytest.raises(
        ComputeError,
        match=(
            r"^Cannot use 'utc=True' with tz-naive data. "
            r"Parse the data as naive, and then use `.dt.convert_time_zone\('UTC'\)`.$"
        ),
    ):
        pl.Series(["2020-01-01 00:00:00"]).str.strptime(pl.Datetime, fmt, utc=True)


@pytest.mark.parametrize(
    ("tz_string", "timedelta"),
    [("+01:00", timedelta(minutes=60)), ("-01:30", timedelta(hours=-1, minutes=-30))],
)
def test_timezone_aware_strptime(tz_string: str, timedelta: timedelta) -> None:
    times = pl.DataFrame(
        {
            "delivery_datetime": [
                "2021-12-05 06:00:00" + tz_string,
                "2021-12-05 07:00:00" + tz_string,
                "2021-12-05 08:00:00" + tz_string,
            ]
        }
    )
    assert times.with_columns(
        pl.col("delivery_datetime").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S%z")
    ).to_dict(False) == {
        "delivery_datetime": [
            datetime(2021, 12, 5, 6, 0, tzinfo=timezone(timedelta)),
            datetime(2021, 12, 5, 7, 0, tzinfo=timezone(timedelta)),
            datetime(2021, 12, 5, 8, 0, tzinfo=timezone(timedelta)),
        ]
    }


def test_non_exact_strptime() -> None:
    s = pl.Series("a", ["2022-01-16", "2022-01-17", "foo2022-01-18", "b2022-01-19ar"])
    fmt = "%Y-%m-%d"

    result = s.str.strptime(pl.Date, fmt, strict=False, exact=True)
    expected = pl.Series("a", [date(2022, 1, 16), date(2022, 1, 17), None, None])
    assert_series_equal(result, expected)

    result = s.str.strptime(pl.Date, fmt, strict=False, exact=False)
    expected = pl.Series(
        "a",
        [date(2022, 1, 16), date(2022, 1, 17), date(2022, 1, 18), date(2022, 1, 19)],
    )
    assert_series_equal(result, expected)

    with pytest.raises(Exception):
        s.str.strptime(pl.Date, fmt, strict=True, exact=True)


def test_strptime_dates_datetimes() -> None:
    s = pl.Series("date", ["2021-04-22", "2022-01-04 00:00:00"])
    assert s.str.strptime(pl.Datetime).to_list() == [
        datetime(2021, 4, 22, 0, 0),
        datetime(2022, 1, 4, 0, 0),
    ]


@pytest.mark.parametrize(
    ("time_string", "expected"),
    [
        ("09-05-2019", datetime(2019, 5, 9)),
        ("2018-09-05", datetime(2018, 9, 5)),
        ("2018-09-05T04:05:01", datetime(2018, 9, 5, 4, 5, 1)),
        ("2018-09-05T04:24:01.9", datetime(2018, 9, 5, 4, 24, 1, 900000)),
        ("2018-09-05T04:24:02.11", datetime(2018, 9, 5, 4, 24, 2, 110000)),
        ("2018-09-05T14:24:02.123", datetime(2018, 9, 5, 14, 24, 2, 123000)),
        ("2019-04-18T02:45:55.555000000", datetime(2019, 4, 18, 2, 45, 55, 555000)),
        ("2019-04-18T22:45:55.555123", datetime(2019, 4, 18, 22, 45, 55, 555123)),
    ],
)
def test_datetime_strptime_patterns_single(time_string: str, expected: str) -> None:
    result = pl.Series([time_string]).str.strptime(pl.Datetime).item()
    assert result == expected


def test_datetime_strptime_patterns_consistent() -> None:
    # note that all should be year first
    df = pl.Series(
        "date",
        [
            "2018-09-05",
            "2018-09-05T04:05:01",
            "2018-09-05T04:24:01.9",
            "2018-09-05T04:24:02.11",
            "2018-09-05T14:24:02.123",
            "2018-09-05T14:24:02.123Z",
            "2019-04-18T02:45:55.555000000",
            "2019-04-18T22:45:55.555123",
        ],
    ).to_frame()
    s = df.with_columns(
        [
            pl.col("date")
            .str.strptime(pl.Datetime, fmt=None, strict=False)
            .alias("parsed"),
        ]
    )["parsed"]
    assert s.null_count() == 1
    assert s[5] is None


def test_datetime_strptime_patterns_inconsistent() -> None:
    # note that the pattern is inferred from the first element to
    # be DatetimeDMY, and so the others (correctly) parse as `null`.
    df = pl.Series(
        "date",
        [
            "09-05-2019",
            "2018-09-05",
            "2018-09-05T04:05:01",
            "2018-09-05T04:24:01.9",
            "2018-09-05T04:24:02.11",
            "2018-09-05T14:24:02.123",
            "2018-09-05T14:24:02.123Z",
            "2019-04-18T02:45:55.555000000",
            "2019-04-18T22:45:55.555123",
        ],
    ).to_frame()
    s = df.with_columns(
        [
            pl.col("date")
            .str.strptime(pl.Datetime, fmt=None, strict=False)
            .alias("parsed"),
        ]
    )["parsed"]
    assert s.null_count() == 8
    assert s[0] is not None


@pytest.mark.parametrize(
    (
        "ts",
        "fmt",
        "exp_year",
        "exp_month",
        "exp_day",
        "exp_hour",
        "exp_minute",
        "exp_second",
    ),
    [
        ("-0031-04-24 22:13:20", "%Y-%m-%d %H:%M:%S", -31, 4, 24, 22, 13, 20),
        ("-0031-04-24", "%Y-%m-%d", -31, 4, 24, 0, 0, 0),
    ],
)
def test_parse_negative_dates(
    ts: str,
    fmt: str,
    exp_year: int,
    exp_month: int,
    exp_day: int,
    exp_hour: int,
    exp_minute: int,
    exp_second: int,
) -> None:
    ser = pl.Series([ts])
    result = ser.str.strptime(pl.Datetime("ms"), fmt=fmt)
    # Python datetime.datetime doesn't support negative dates, so comparing
    # with `result.item()` directly won't work.
    assert result.dt.year().item() == exp_year
    assert result.dt.month().item() == exp_month
    assert result.dt.day().item() == exp_day
    assert result.dt.hour().item() == exp_hour
    assert result.dt.minute().item() == exp_minute
    assert result.dt.second().item() == exp_second


def test_short_formats() -> None:
    s = pl.Series(["20202020", "2020"])
    assert s.str.strptime(pl.Date, "%Y", strict=False).to_list() == [
        None,
        date(2020, 1, 1),
    ]
    assert s.str.strptime(pl.Date, "%foo", strict=False).to_list() == [None, None]


@pytest.mark.parametrize(
    ("time_string", "fmt", "datatype", "expected"),
    [
        ("Jul/2020", "%b/%Y", pl.Date, date(2020, 7, 1)),
        ("Jan/2020", "%b/%Y", pl.Date, date(2020, 1, 1)),
        ("02/Apr/2020", "%d/%b/%Y", pl.Date, date(2020, 4, 2)),
        ("Dec/2020", "%b/%Y", pl.Datetime, datetime(2020, 12, 1, 0, 0)),
        ("Nov/2020", "%b/%Y", pl.Datetime, datetime(2020, 11, 1, 0, 0)),
        ("02/Feb/2020", "%d/%b/%Y", pl.Datetime, datetime(2020, 2, 2, 0, 0)),
    ],
)
def test_abbrev_month(
    time_string: str, fmt: str, datatype: PolarsTemporalType, expected: date
) -> None:
    s = pl.Series([time_string])
    result = s.str.strptime(datatype, fmt).item()
    assert result == expected


def test_invalid_date_parsing_4898() -> None:
    assert pl.Series(["2022-09-18", "2022-09-50"]).str.strptime(
        pl.Date, "%Y-%m-%d", strict=False
    ).to_list() == [date(2022, 9, 18), None]


def test_replace_timezone_invalid_timezone() -> None:
    ts = pl.Series(["2020-01-01 00:00:00+01:00"]).str.strptime(
        pl.Datetime, "%Y-%m-%d %H:%M:%S%z"
    )
    with pytest.raises(ComputeError, match=r"Could not parse time zone foo"):
        ts.dt.replace_time_zone("foo")


@pytest.mark.parametrize(
    ("ts", "fmt", "expected"),
    [
        ("2020-01-01T00:00:00Z", "%+", datetime(2020, 1, 1, tzinfo=timezone.utc)),
        (
            "2020-01-01T00:00:00+01:00",
            "%Y-%m-%dT%H:%M:%S%z",
            datetime(2020, 1, 1, tzinfo=timezone(timedelta(seconds=3600))),
        ),
        (
            "2020-01-01T00:00:00+01:00",
            "%Y-%m-%dT%H:%M:%S%:z",
            datetime(2020, 1, 1, tzinfo=timezone(timedelta(seconds=3600))),
        ),
        (
            "2020-01-01T00:00:00+01:00",
            "%Y-%m-%dT%H:%M:%S%#z",
            datetime(2020, 1, 1, tzinfo=timezone(timedelta(seconds=3600))),
        ),
    ],
)
def test_tz_aware_strptime(ts: str, fmt: str, expected: datetime) -> None:
    result = pl.Series([ts]).str.strptime(pl.Datetime, fmt).item()
    assert result == expected


@pytest.mark.parametrize("fmt", ["%+", "%Y-%m-%dT%H:%M:%S%z"])
def test_crossing_dst(fmt: str) -> None:
    ts = ["2021-03-27T23:59:59+01:00", "2021-03-28T23:59:59+02:00"]
    result = pl.Series(ts).str.strptime(pl.Datetime, fmt, utc=True)
    assert result[0] == datetime(2021, 3, 27, 22, 59, 59, tzinfo=ZoneInfo(key="UTC"))
    assert result[1] == datetime(2021, 3, 28, 21, 59, 59, tzinfo=ZoneInfo(key="UTC"))


@pytest.mark.parametrize("fmt", ["%+", "%Y-%m-%dT%H:%M:%S%z"])
def test_crossing_dst_tz_aware(fmt: str) -> None:
    ts = ["2021-03-27T23:59:59+01:00", "2021-03-28T23:59:59+02:00"]
    with pytest.raises(
        ComputeError,
        match=(
            r"^Different timezones found during 'strptime' operation. "
            "You might want to use `utc=True` and then set the time zone after parsing$"
        ),
    ):
        pl.Series(ts).str.strptime(pl.Datetime, fmt, utc=False)


def test_tz_aware_without_fmt() -> None:
    with pytest.raises(
        ComputeError,
        match=(
            r"^Passing 'tz_aware=True' without 'fmt' is not yet supported. "
            r"Please specify 'fmt'.$"
        ),
    ):
        pl.Series(["2020-01-01"]).str.strptime(pl.Datetime, tz_aware=True)
