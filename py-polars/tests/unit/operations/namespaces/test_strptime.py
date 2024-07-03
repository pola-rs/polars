"""
Module for testing `.str.strptime` of the string namespace.

This method gets its own module due to its complexity.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ChronoFormatWarning, ComputeError, InvalidOperationError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars._typing import PolarsTemporalType, TimeUnit
else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


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
    assert df.select(pl.col("month").str.to_date(format="%Y-%m")).item() == date(
        2022, 1, 1
    )
    assert df.select(
        pl.col("month").str.to_datetime(format="%Y-%m")
    ).item() == datetime(2022, 1, 1)


def test_to_datetime_precision() -> None:
    s = pl.Series(
        "date", ["2022-09-12 21:54:36.789321456", "2022-09-13 12:34:56.987456321"]
    )
    ds = s.str.to_datetime()
    assert ds.cast(pl.Date).is_not_null().all()
    assert getattr(ds.dtype, "time_unit", None) == "us"

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
    for time_unit, suffix, expected_values in test_data:
        ds = s.str.to_datetime(f"%Y-%m-%d %H:%M:%S{suffix}", time_unit=time_unit)
        assert getattr(ds.dtype, "time_unit", None) == time_unit
        assert ds.dt.nanosecond().to_list() == expected_values


@pytest.mark.parametrize(
    ("time_unit", "expected"),
    [("ms", "123000000"), ("us", "123456000"), ("ns", "123456789")],
)
@pytest.mark.parametrize("format", ["%Y-%m-%d %H:%M:%S%.f", None])
def test_to_datetime_precision_with_time_unit(
    time_unit: TimeUnit, expected: str, format: str
) -> None:
    s = pl.Series(["2020-01-01 00:00:00.123456789"])
    result = s.str.to_datetime(format, time_unit=time_unit).dt.to_string("%f")[0]
    assert result == expected


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
        pl.col("delivery_datetime").str.to_datetime(format="%Y-%m-%d %H:%M:%S%z")
    ).to_dict(as_series=False) == {
        "delivery_datetime": [
            datetime(2021, 12, 5, 6, 0, tzinfo=timezone(timedelta)),
            datetime(2021, 12, 5, 7, 0, tzinfo=timezone(timedelta)),
            datetime(2021, 12, 5, 8, 0, tzinfo=timezone(timedelta)),
        ]
    }


def test_to_date_non_exact_strptime() -> None:
    s = pl.Series("a", ["2022-01-16", "2022-01-17", "foo2022-01-18", "b2022-01-19ar"])
    format = "%Y-%m-%d"

    result = s.str.to_date(format, strict=False, exact=True)
    expected = pl.Series("a", [date(2022, 1, 16), date(2022, 1, 17), None, None])
    assert_series_equal(result, expected)

    result = s.str.to_date(format, strict=False, exact=False)
    expected = pl.Series(
        "a",
        [date(2022, 1, 16), date(2022, 1, 17), date(2022, 1, 18), date(2022, 1, 19)],
    )
    assert_series_equal(result, expected)

    with pytest.raises(InvalidOperationError):
        s.str.to_date(format, strict=True, exact=True)


@pytest.mark.parametrize(
    ("time_string", "expected"),
    [
        ("01-02-2024", date(2024, 2, 1)),
        ("01.02.2024", date(2024, 2, 1)),
        ("01/02/2024", date(2024, 2, 1)),
        ("2024-02-01", date(2024, 2, 1)),
        ("2024/02/01", date(2024, 2, 1)),
        ("31-12-2024", date(2024, 12, 31)),
        ("31.12.2024", date(2024, 12, 31)),
        ("31/12/2024", date(2024, 12, 31)),
        ("2024-12-31", date(2024, 12, 31)),
        ("2024/12/31", date(2024, 12, 31)),
    ],
)
def test_to_date_all_inferred_date_patterns(time_string: str, expected: date) -> None:
    result = pl.Series([time_string]).str.to_date()
    assert result[0] == expected


@pytest.mark.parametrize(
    ("value", "attr"),
    [
        ("a", "to_date"),
        ("ab", "to_date"),
        ("a", "to_datetime"),
        ("ab", "to_datetime"),
    ],
)
def test_non_exact_short_elements_10223(value: str, attr: str) -> None:
    with pytest.raises(InvalidOperationError, match="conversion .* failed"):
        getattr(pl.Series(["2019-01-01", value]).str, attr)(exact=False)


@pytest.mark.parametrize(
    ("offset", "time_zone", "tzinfo", "format"),
    [
        ("+01:00", "UTC", timezone(timedelta(hours=1)), "%Y-%m-%dT%H:%M%z"),
        ("", None, None, "%Y-%m-%dT%H:%M"),
    ],
)
def test_to_datetime_non_exact_strptime(
    offset: str, time_zone: str | None, tzinfo: timezone | None, format: str
) -> None:
    s = pl.Series(
        "a",
        [
            f"2022-01-16T00:00{offset}",
            f"2022-01-17T00:00{offset}",
            f"foo2022-01-18T00:00{offset}",
            f"b2022-01-19T00:00{offset}ar",
        ],
    )

    result = s.str.to_datetime(format, strict=False, exact=True)
    expected = pl.Series(
        "a",
        [
            datetime(2022, 1, 16, tzinfo=tzinfo),
            datetime(2022, 1, 17, tzinfo=tzinfo),
            None,
            None,
        ],
    )
    assert_series_equal(result, expected)
    assert result.dtype == pl.Datetime("us", time_zone)

    result = s.str.to_datetime(format, strict=False, exact=False)
    expected = pl.Series(
        "a",
        [
            datetime(2022, 1, 16, tzinfo=tzinfo),
            datetime(2022, 1, 17, tzinfo=tzinfo),
            datetime(2022, 1, 18, tzinfo=tzinfo),
            datetime(2022, 1, 19, tzinfo=tzinfo),
        ],
    )
    assert_series_equal(result, expected)
    assert result.dtype == pl.Datetime("us", time_zone)

    with pytest.raises(InvalidOperationError):
        s.str.to_datetime(format, strict=True, exact=True)


def test_to_datetime_dates_datetimes() -> None:
    s = pl.Series("date", ["2021-04-22", "2022-01-04 00:00:00"])
    assert s.str.to_datetime().to_list() == [
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
        (
            "2018-09-05T04:05:01+01:00",
            datetime(2018, 9, 5, 4, 5, 1, tzinfo=timezone(timedelta(hours=1))),
        ),
        (
            "2018-09-05T04:24:01.9+01:00",
            datetime(2018, 9, 5, 4, 24, 1, 900000, tzinfo=timezone(timedelta(hours=1))),
        ),
        (
            "2018-09-05T04:24:02.11+01:00",
            datetime(2018, 9, 5, 4, 24, 2, 110000, tzinfo=timezone(timedelta(hours=1))),
        ),
        (
            "2018-09-05T14:24:02.123+01:00",
            datetime(
                2018, 9, 5, 14, 24, 2, 123000, tzinfo=timezone(timedelta(hours=1))
            ),
        ),
        (
            "2019-04-18T02:45:55.555000000+01:00",
            datetime(
                2019, 4, 18, 2, 45, 55, 555000, tzinfo=timezone(timedelta(hours=1))
            ),
        ),
        (
            "2019-04-18T22:45:55.555123+01:00",
            datetime(
                2019, 4, 18, 22, 45, 55, 555123, tzinfo=timezone(timedelta(hours=1))
            ),
        ),
    ],
)
def test_to_datetime_patterns_single(time_string: str, expected: str) -> None:
    result = pl.Series([time_string]).str.to_datetime().item()
    assert result == expected


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_infer_tz_aware_time_unit(time_unit: TimeUnit) -> None:
    result = pl.Series(["2020-01-02T04:00:00+02:00"]).str.to_datetime(
        time_unit=time_unit
    )
    assert result.dtype == pl.Datetime(time_unit, "UTC")
    assert result.item() == datetime(2020, 1, 2, 2, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_infer_tz_aware_with_utc(time_unit: TimeUnit) -> None:
    result = pl.Series(["2020-01-02T04:00:00+02:00"]).str.to_datetime(
        time_unit=time_unit
    )
    assert result.dtype == pl.Datetime(time_unit, "UTC")
    assert result.item() == datetime(2020, 1, 2, 2, 0, tzinfo=timezone.utc)


def test_str_to_datetime_infer_tz_aware() -> None:
    result = (
        pl.Series(["2020-01-02T04:00:00+02:00"])
        .str.to_datetime(time_unit="us", time_zone="Europe/Vienna")
        .item()
    )
    assert result == datetime(2020, 1, 2, 3, tzinfo=ZoneInfo("Europe/Vienna"))


@pytest.mark.parametrize(
    "result",
    [
        pl.Series(["2020-01-01T00:00:00+00:00"]).str.strptime(
            pl.Datetime("us", "UTC"), format="%Y-%m-%dT%H:%M:%S%z"
        ),
        pl.Series(["2020-01-01T00:00:00+00:00"]).str.strptime(
            pl.Datetime("us"), format="%Y-%m-%dT%H:%M:%S%z"
        ),
        pl.Series(["2020-01-01T00:00:00+00:00"]).str.strptime(pl.Datetime("us", "UTC")),
        pl.Series(["2020-01-01T00:00:00+00:00"]).str.strptime(pl.Datetime("us")),
        pl.Series(["2020-01-01T00:00:00+00:00"]).str.to_datetime(
            time_zone="UTC", format="%Y-%m-%dT%H:%M:%S%z"
        ),
        pl.Series(["2020-01-01T00:00:00+00:00"]).str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S%z"
        ),
        pl.Series(["2020-01-01T00:00:00+00:00"]).str.to_datetime(time_zone="UTC"),
        pl.Series(["2020-01-01T00:00:00+00:00"]).str.to_datetime(),
    ],
)
def test_parsing_offset_aware_with_utc_dtype(result: pl.Series) -> None:
    expected = pl.Series([datetime(2020, 1, 1, tzinfo=timezone.utc)])
    assert_series_equal(result, expected)


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
        pl.col("date").str.to_datetime(strict=False).alias("parsed"),
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
    s = df.with_columns(pl.col("date").str.to_datetime(strict=False).alias("parsed"))[
        "parsed"
    ]
    assert s.null_count() == 8
    assert s[0] is not None


@pytest.mark.parametrize(
    (
        "ts",
        "format",
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
    format: str,
    exp_year: int,
    exp_month: int,
    exp_day: int,
    exp_hour: int,
    exp_minute: int,
    exp_second: int,
) -> None:
    s = pl.Series([ts])
    result = s.str.to_datetime(format, time_unit="ms")
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
    assert s.str.to_date("%Y", strict=False).to_list() == [
        None,
        date(2020, 1, 1),
    ]
    assert s.str.to_date("%bar", strict=False).to_list() == [None, None]


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
def test_strptime_abbrev_month(
    time_string: str, fmt: str, datatype: PolarsTemporalType, expected: date
) -> None:
    s = pl.Series([time_string])
    result = s.str.strptime(datatype, fmt).item()
    assert result == expected


def test_full_month_name() -> None:
    s = pl.Series(["2022-December-01"]).str.to_datetime("%Y-%B-%d")
    assert s[0] == datetime(2022, 12, 1)


@pytest.mark.parametrize(
    ("datatype", "expected"),
    [
        (pl.Datetime, datetime(2022, 1, 1)),
        (pl.Date, date(2022, 1, 1)),
    ],
)
def test_single_digit_month(
    datatype: PolarsTemporalType, expected: datetime | date
) -> None:
    s = pl.Series(["2022-1-1"]).str.strptime(datatype, "%Y-%m-%d")
    assert s[0] == expected


def test_invalid_date_parsing_4898() -> None:
    assert pl.Series(["2022-09-18", "2022-09-50"]).str.to_date(
        "%Y-%m-%d", strict=False
    ).to_list() == [date(2022, 9, 18), None]


def test_strptime_invalid_timezone() -> None:
    ts = pl.Series(["2020-01-01 00:00:00+01:00"]).str.to_datetime("%Y-%m-%d %H:%M:%S%z")
    with pytest.raises(ComputeError, match=r"unable to parse time zone: 'foo'"):
        ts.dt.replace_time_zone("foo")


def test_to_datetime_ambiguous_or_non_existent() -> None:
    with pytest.raises(
        ComputeError,
        match="datetime '2021-11-07 01:00:00' is ambiguous in time zone 'US/Central'",
    ):
        pl.Series(["2021-11-07 01:00"]).str.to_datetime(
            time_unit="us", time_zone="US/Central"
        )
    with pytest.raises(
        ComputeError,
        match="datetime '2021-03-28 02:30:00' is non-existent in time zone 'Europe/Warsaw'",
    ):
        pl.Series(["2021-03-28 02:30"]).str.to_datetime(
            time_unit="us", time_zone="Europe/Warsaw"
        )
    with pytest.raises(
        ComputeError,
        match="datetime '2021-03-28 02:30:00' is non-existent in time zone 'Europe/Warsaw'",
    ):
        pl.Series(["2021-03-28 02:30"]).str.to_datetime(
            time_unit="us",
            time_zone="Europe/Warsaw",
            ambiguous="null",
        )
    with pytest.raises(
        ComputeError,
        match="datetime '2021-03-28 02:30:00' is non-existent in time zone 'Europe/Warsaw'",
    ):
        pl.Series(["2021-03-28 02:30"] * 2).str.to_datetime(
            time_unit="us",
            time_zone="Europe/Warsaw",
            ambiguous=pl.Series(["null", "null"]),
        )


@pytest.mark.parametrize(
    ("ts", "fmt", "expected"),
    [
        ("2020-01-01T00:00:00Z", None, datetime(2020, 1, 1, tzinfo=timezone.utc)),
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
def test_to_datetime_tz_aware_strptime(ts: str, fmt: str, expected: datetime) -> None:
    result = pl.Series([ts]).str.to_datetime(fmt).item()
    assert result == expected


@pytest.mark.parametrize("format", ["%+", "%Y-%m-%dT%H:%M:%S%z"])
def test_crossing_dst(format: str) -> None:
    ts = ["2021-03-27T23:59:59+01:00", "2021-03-28T23:59:59+02:00"]
    result = pl.Series(ts).str.to_datetime(format)
    assert result[0] == datetime(2021, 3, 27, 22, 59, 59, tzinfo=ZoneInfo("UTC"))
    assert result[1] == datetime(2021, 3, 28, 21, 59, 59, tzinfo=ZoneInfo("UTC"))


@pytest.mark.parametrize("format", ["%+", "%Y-%m-%dT%H:%M:%S%z"])
def test_crossing_dst_tz_aware(format: str) -> None:
    ts = ["2021-03-27T23:59:59+01:00", "2021-03-28T23:59:59+02:00"]
    result = pl.Series(ts).str.to_datetime(format)
    expected = pl.Series(
        [
            datetime(2021, 3, 27, 22, 59, 59, tzinfo=timezone.utc),
            datetime(2021, 3, 28, 21, 59, 59, tzinfo=timezone.utc),
        ]
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "format", "expected"),
    [
        (
            "2023-02-05T05:10:10.074000",
            "%Y-%m-%dT%H:%M:%S%.f",
            datetime(2023, 2, 5, 5, 10, 10, 74000),
        ),
    ],
)
def test_strptime_subseconds_datetime(data: str, format: str, expected: time) -> None:
    s = pl.Series([data])
    result = s.str.to_datetime(format).item()
    assert result == expected


@pytest.mark.parametrize(
    ("string", "fmt"),
    [
        pytest.param("2023-05-04|7", "%Y-%m-%d|%H", id="hour but no minute"),
        pytest.param("2023-05-04|7", "%Y-%m-%d|%k", id="padded hour but no minute"),
        pytest.param("2023-05-04|10", "%Y-%m-%d|%M", id="minute but no hour"),
        pytest.param("2023-05-04|10", "%Y-%m-%d|%S", id="second but no hour"),
        pytest.param(
            "2000-Jan-01 01 00 01", "%Y-%b-%d %I %M %S", id="12-hour clock but no AM/PM"
        ),
        pytest.param(
            "2000-Jan-01 01 00 01",
            "%Y-%b-%d %l %M %S",
            id="padded 12-hour clock but no AM/PM",
        ),
    ],
)
def test_strptime_incomplete_formats(string: str, fmt: str) -> None:
    with pytest.raises(
        ComputeError,
        match="Invalid format string",
    ):
        pl.Series([string]).str.to_datetime(fmt)


@pytest.mark.parametrize(
    ("string", "fmt", "expected"),
    [
        ("2023-05-04|7:3", "%Y-%m-%d|%H:%M", datetime(2023, 5, 4, 7, 3)),
        ("2023-05-04|10:03", "%Y-%m-%d|%H:%M", datetime(2023, 5, 4, 10, 3)),
        (
            "2000-Jan-01 01 00 01 am",
            "%Y-%b-%d %I %M %S %P",
            datetime(2000, 1, 1, 1, 0, 1),
        ),
        (
            "2000-Jan-01 01 00 01 am",
            "%Y-%b-%d %_I %M %S %P",
            datetime(2000, 1, 1, 1, 0, 1),
        ),
        (
            "2000-Jan-01 01 00 01 am",
            "%Y-%b-%d %l %M %S %P",
            datetime(2000, 1, 1, 1, 0, 1),
        ),
        (
            "2000-Jan-01 01 00 01 AM",
            "%Y-%b-%d %I %M %S %p",
            datetime(2000, 1, 1, 1, 0, 1),
        ),
        (
            "2000-Jan-01 01 00 01 AM",
            "%Y-%b-%d %_I %M %S %p",
            datetime(2000, 1, 1, 1, 0, 1),
        ),
        (
            "2000-Jan-01 01 00 01 AM",
            "%Y-%b-%d %l %M %S %p",
            datetime(2000, 1, 1, 1, 0, 1),
        ),
    ],
)
def test_strptime_complete_formats(string: str, fmt: str, expected: datetime) -> None:
    # Similar to the above, but these formats are complete and should work
    result = pl.Series([string]).str.to_datetime(fmt).item()
    assert result == expected


@pytest.mark.parametrize(
    ("data", "format", "expected"),
    [
        ("05:10:10.074000", "%H:%M:%S%.f", time(5, 10, 10, 74000)),
        ("05:10:10.074000", "%T%.6f", time(5, 10, 10, 74000)),
        ("05:10:10.074000", "%H:%M:%S%.3f", time(5, 10, 10, 74000)),
    ],
)
def test_to_time_subseconds(data: str, format: str, expected: time) -> None:
    s = pl.Series([data])
    for res in (
        s.str.to_time().item(),
        s.str.to_time(format).item(),
    ):
        assert res == expected


def test_to_time_format_warning() -> None:
    s = pl.Series(["05:10:10.074000"])
    with pytest.warns(ChronoFormatWarning, match=".%f"):
        result = s.str.to_time("%H:%M:%S.%f").item()
    assert result == time(5, 10, 10, 74)


@pytest.mark.parametrize("exact", [True, False])
def test_to_datetime_ambiguous_earliest(exact: bool) -> None:
    result = (
        pl.Series(["2020-10-25 01:00"])
        .str.to_datetime(time_zone="Europe/London", ambiguous="earliest", exact=exact)
        .item()
    )
    expected = datetime(2020, 10, 25, 1, fold=0, tzinfo=ZoneInfo("Europe/London"))
    assert result == expected
    result = (
        pl.Series(["2020-10-25 01:00"])
        .str.to_datetime(time_zone="Europe/London", ambiguous="latest", exact=exact)
        .item()
    )
    expected = datetime(2020, 10, 25, 1, fold=1, tzinfo=ZoneInfo("Europe/London"))
    assert result == expected
    with pytest.raises(ComputeError):
        pl.Series(["2020-10-25 01:00"]).str.to_datetime(
            time_zone="Europe/London",
            exact=exact,
        ).item()


def test_to_datetime_naive_format_and_time_zone() -> None:
    # format-specified path
    result = pl.Series(["2020-01-01"]).str.to_datetime(
        format="%Y-%m-%d", time_zone="Asia/Kathmandu"
    )
    expected = pl.Series([datetime(2020, 1, 1)]).dt.replace_time_zone("Asia/Kathmandu")
    assert_series_equal(result, expected)
    # format-inferred path
    result = pl.Series(["2020-01-01"]).str.to_datetime(time_zone="Asia/Kathmandu")
    assert_series_equal(result, expected)


@pytest.mark.parametrize("exact", [True, False])
def test_strptime_ambiguous_earliest(exact: bool) -> None:
    result = (
        pl.Series(["2020-10-25 01:00"])
        .str.strptime(
            pl.Datetime("us", "Europe/London"), ambiguous="earliest", exact=exact
        )
        .item()
    )
    expected = datetime(2020, 10, 25, 1, fold=0, tzinfo=ZoneInfo("Europe/London"))
    assert result == expected
    result = (
        pl.Series(["2020-10-25 01:00"])
        .str.strptime(
            pl.Datetime("us", "Europe/London"), ambiguous="latest", exact=exact
        )
        .item()
    )
    expected = datetime(2020, 10, 25, 1, fold=1, tzinfo=ZoneInfo("Europe/London"))
    assert result == expected
    with pytest.raises(ComputeError):
        pl.Series(["2020-10-25 01:00"]).str.strptime(
            pl.Datetime("us", "Europe/London"),
            exact=exact,
        ).item()


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_to_datetime_out_of_range_13401(time_unit: TimeUnit) -> None:
    s = pl.Series(["2020-January-01 12:34:66"])
    with pytest.raises(InvalidOperationError, match="conversion .* failed"):
        s.str.to_datetime("%Y-%B-%d %H:%M:%S", time_unit=time_unit)
    assert (
        s.str.to_datetime("%Y-%B-%d %H:%M:%S", strict=False, time_unit=time_unit).item()
        is None
    )


def test_out_of_ns_range_no_tu_specified_13592() -> None:
    df = pl.DataFrame({"dates": ["2022-08-31 00:00:00.0", "0920-09-18 00:00:00.0"]})
    result = df.select(pl.col("dates").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f"))[
        "dates"
    ]
    expected = pl.Series(
        "dates",
        [datetime(2022, 8, 31, 0, 0), datetime(920, 9, 18, 0, 0)],
        dtype=pl.Datetime("us"),
    )
    assert_series_equal(result, expected)
