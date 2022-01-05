from datetime import date, datetime, timedelta

from polars.utils import (
    _date_to_pl_date,
    _datetime_to_pl_timestamp,
    _timedelta_to_pl_timedelta,
    in_nanoseconds_window,
)


def test_in_ns_window() -> None:
    assert not in_nanoseconds_window(datetime(year=2600, month=1, day=1))
    assert in_nanoseconds_window(datetime(year=2000, month=1, day=1))


def test_datetime_to_pl_timestamp() -> None:
    out = _datetime_to_pl_timestamp(datetime(2121, 1, 1), "ns")
    assert out == 4765132800000000000
    out = _datetime_to_pl_timestamp(datetime(2121, 1, 1), "ms")
    assert out == 4765132800000


def test_date_to_pl_date() -> None:
    d = date(1999, 9, 9)
    out = _date_to_pl_date(d)
    assert out == 10843


def test_timedelta_to_pl_timedelta() -> None:
    out = _timedelta_to_pl_timedelta(timedelta(days=1), "ns")
    assert out == 86_400_000_000_000
    out = _timedelta_to_pl_timedelta(timedelta(days=1), "ms")
    assert out == 86_400_000
    out = _timedelta_to_pl_timedelta(timedelta(days=1), tu=None)
    assert out == 86_400_000_000_000
