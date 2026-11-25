from __future__ import annotations

import calendar
from datetime import date, datetime, time
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit

VALUES: list[datetime | None] = [
    datetime(1970, 1, 1),
    datetime(1969, 12, 31, 23, 59, 59, 999999),
    datetime(1969, 12, 31, 0, 0, 0, 1),
    datetime(1900, 2, 28, 13, 14, 15, 123456),
    datetime(1600, 2, 29, 23, 59, 59, 999000),
    datetime(1, 1, 1, 0, 0, 1),
    None,
    datetime(2000, 2, 29, 12, 30, 45, 678000),
    datetime(2024, 12, 31, 23, 59, 59, 999000),
    datetime(2100, 3, 1, 5, 6, 7, 8000),
]


def _expected_time_of_day(v: datetime | None) -> dict[str, int | None]:
    if v is None:
        return dict.fromkeys(["h", "mi", "s", "ms", "us", "ns"])
    return {
        "h": v.hour,
        "mi": v.minute,
        "s": v.second,
        "ms": v.microsecond // 1000,
        "us": v.microsecond,
        "ns": v.microsecond * 1000,
    }


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("time_zone", [None, "UTC"])
def test_time_of_day_fields(time_unit: TimeUnit, time_zone: str | None) -> None:
    if time_unit == "ns":
        values = [v for v in VALUES if v is None or v.year >= 1678]
    else:
        values = VALUES
    df = pl.DataFrame({"a": values}, schema={"a": pl.Datetime(time_unit, time_zone)})
    out = df.select(
        h=pl.col("a").dt.hour(),
        mi=pl.col("a").dt.minute(),
        s=pl.col("a").dt.second(),
        ms=pl.col("a").dt.millisecond(),
        us=pl.col("a").dt.microsecond(),
        ns=pl.col("a").dt.nanosecond(),
    )
    expected = pl.DataFrame(
        [_expected_time_of_day(v) for v in df["a"].to_list()],
        schema={
            "h": pl.Int8,
            "mi": pl.Int8,
            "s": pl.Int8,
            "ms": pl.Int32,
            "us": pl.Int32,
            "ns": pl.Int32,
        },
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
def test_time_of_day_fields_utc_matches_other_time_zone(time_unit: TimeUnit) -> None:
    values = [v for v in VALUES if v is None or v.year >= 1678]
    naive = pl.DataFrame({"a": values}, schema={"a": pl.Datetime(time_unit)})
    expr = [
        pl.col("a").dt.hour(),
        pl.col("a").dt.minute().alias("mi"),
        pl.col("a").dt.second().alias("s"),
        pl.col("a").dt.millisecond().alias("ms"),
        pl.col("a").dt.microsecond().alias("us"),
        pl.col("a").dt.nanosecond().alias("ns"),
        pl.col("a").dt.year().alias("y"),
        pl.col("a").dt.month().alias("mo"),
        pl.col("a").dt.day().alias("d"),
    ]
    fixed = naive.with_columns(pl.col("a").dt.replace_time_zone("Etc/GMT+0"))
    utc = naive.with_columns(pl.col("a").dt.replace_time_zone("UTC"))
    assert_frame_equal(utc.select(expr), fixed.select(expr))
    assert_frame_equal(naive.select(expr), fixed.select(expr))


def test_time_of_day_fields_out_of_range_are_null() -> None:
    s = pl.Series([2**62, -(2**62), 0], dtype=pl.Int64).cast(pl.Datetime("ms"))
    out = pl.select(
        h=s.dt.hour(), ms=s.dt.millisecond(), y=s.dt.year(), mo=s.dt.month()
    )
    assert out["h"].to_list() == [None, None, 0]
    assert out["ms"].to_list() == [None, None, 0]
    assert out["y"].to_list() == [None, None, 1970]


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
def test_datetime_to_date_and_time(time_unit: TimeUnit) -> None:
    values = [v for v in VALUES if v is None or v.year >= 1678]
    df = pl.DataFrame({"a": values}, schema={"a": pl.Datetime(time_unit)})
    out = df.select(d=pl.col("a").dt.date(), t=pl.col("a").dt.time())
    assert out.schema == {"d": pl.Date, "t": pl.Time}
    expected_dates = [None if v is None else v.date() for v in df["a"].to_list()]
    expected_times = [None if v is None else v.time() for v in df["a"].to_list()]
    assert out["d"].to_list() == expected_dates
    assert out["t"].to_list() == expected_times


def test_datetime_to_date_overflow() -> None:
    s = pl.Series([2**62, 0, None], dtype=pl.Int64).cast(pl.Datetime("ms"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        s.cast(pl.Date)
    assert s.cast(pl.Date, strict=False).to_list() == [None, date(1970, 1, 1), None]
    assert s.dt.time().to_list() == [time(15, 36, 27, 904000), time(0), None]


def test_weekday_all_time_units() -> None:
    for tu in ("ns", "us", "ms"):
        df = pl.DataFrame(
            {"a": [v for v in VALUES if v is None or v.year >= 1678]},
            schema={"a": pl.Datetime(tu)},  # type: ignore[arg-type]
        )
        out = df["a"].dt.weekday().to_list()
        assert out == [None if v is None else v.isoweekday() for v in df["a"].to_list()]


def _month_bounds(v: date) -> tuple[date, date]:
    last = calendar.monthrange(v.year, v.month)[1]
    return v.replace(day=1), v.replace(day=last)


DATES = [
    date(1970, 1, 1),
    date(1969, 12, 31),
    date(1969, 1, 15),
    date(1900, 2, 28),
    date(1900, 3, 1),
    date(1600, 2, 29),
    date(2000, 2, 1),
    date(2000, 2, 29),
    date(2023, 2, 28),
    date(2024, 2, 10),
    date(2024, 12, 31),
    date(2100, 2, 28),
    date(2100, 12, 1),
    date(1, 1, 1),
    date(9999, 12, 31),
]


def test_month_start_end_date() -> None:
    s = pl.Series(DATES + [None], dtype=pl.Date)
    expected_start = [None if v is None else _month_bounds(v)[0] for v in s]
    expected_end = [None if v is None else _month_bounds(v)[1] for v in s]
    assert s.dt.month_start().to_list() == expected_start
    assert s.dt.month_end().to_list() == expected_end


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("time_zone", [None, "UTC"])
def test_month_start_end_datetime(time_unit: TimeUnit, time_zone: str | None) -> None:
    values = [v for v in VALUES if v is None or v.year >= 1678]
    values += [datetime(2024, 2, 29, 23, 59, 59, 999000), datetime(2023, 2, 1, 0, 0, 1)]
    s = pl.Series(values, dtype=pl.Datetime(time_unit, time_zone))
    expected_start = []
    expected_end = []
    for v in values:
        if v is None:
            expected_start.append(None)
            expected_end.append(None)
            continue
        first, last = _month_bounds(v.date())
        expected_start.append(datetime.combine(first, v.time()))
        expected_end.append(datetime.combine(last, v.time()))
    start = pl.Series(expected_start, dtype=pl.Datetime(time_unit, time_zone))
    end = pl.Series(expected_end, dtype=pl.Datetime(time_unit, time_zone))
    assert s.dt.month_start().to_list() == start.to_list()
    assert s.dt.month_end().to_list() == end.to_list()


def test_month_end_datetime_near_max_ns() -> None:
    s = pl.Series([datetime(2262, 4, 1)], dtype=pl.Datetime("ns"))
    assert s.dt.month_start().to_list() == [datetime(2262, 4, 1)]
