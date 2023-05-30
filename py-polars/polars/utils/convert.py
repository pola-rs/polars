from __future__ import annotations

import sys
from datetime import datetime, time, timedelta, timezone
from decimal import Context
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, overload

from polars.dependencies import _ZONEINFO_AVAILABLE, zoneinfo

if TYPE_CHECKING:
    from collections.abc import Reversible
    from datetime import date, tzinfo
    from decimal import Decimal

    from polars.type_aliases import TimeUnit

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")

    # the below shenanigans with ZoneInfo are all to handle a
    # typing issue in py < 3.9 while preserving lazy-loading
    if sys.version_info >= (3, 9):
        from zoneinfo import ZoneInfo
    elif _ZONEINFO_AVAILABLE:
        from backports.zoneinfo._zoneinfo import ZoneInfo

    def get_zoneinfo(key: str) -> ZoneInfo:
        pass

else:

    @lru_cache(None)
    def get_zoneinfo(key: str) -> ZoneInfo:
        return zoneinfo.ZoneInfo(key)


# note: reversed views don't match as instances of MappingView
if sys.version_info >= (3, 11):
    _views: list[Reversible[Any]] = [{}.keys(), {}.values(), {}.items()]
    _reverse_mapping_views = tuple(type(reversed(view)) for view in _views)


@overload
def _timedelta_to_pl_duration(td: None) -> None:
    ...


@overload
def _timedelta_to_pl_duration(td: timedelta | str) -> str:
    ...


def _timedelta_to_pl_duration(td: timedelta | str | None) -> str | None:
    """Convert python timedelta to a polars duration string."""
    if td is None or isinstance(td, str):
        return td
    else:
        if td.days >= 0:
            d = td.days and f"{td.days}d" or ""
            s = td.seconds and f"{td.seconds}s" or ""
            us = td.microseconds and f"{td.microseconds}us" or ""
        else:
            if not td.seconds and not td.microseconds:
                d = td.days and f"{td.days}d" or ""
                s = ""
                us = ""
            else:
                corrected_d = td.days + 1
                d = corrected_d and f"{corrected_d}d" or "-"
                corrected_seconds = 24 * 3600 - (td.seconds + (td.microseconds > 0))
                s = corrected_seconds and f"{corrected_seconds}s" or ""
                us = td.microseconds and f"{10**6 - td.microseconds}us" or ""

        return f"{d}{s}{us}"


def _datetime_to_pl_timestamp(dt: datetime, time_unit: TimeUnit | None) -> int:
    """Convert a python datetime to a timestamp in nanoseconds."""
    dt = dt.replace(tzinfo=timezone.utc)
    if time_unit == "ns":
        nanos = dt.microsecond * 1000
        return int(dt.timestamp()) * 1_000_000_000 + nanos
    elif time_unit == "us":
        micros = dt.microsecond
        return int(dt.timestamp()) * 1_000_000 + micros
    elif time_unit == "ms":
        millis = dt.microsecond // 1000
        return int(dt.timestamp()) * 1_000 + millis
    elif time_unit is None:
        # python has us precision
        micros = dt.microsecond
        return int(dt.timestamp()) * 1_000_000 + micros
    else:
        raise ValueError(
            f"time_unit must be one of {{'ns', 'us', 'ms'}}, got {time_unit}"
        )


def _time_to_pl_time(t: time) -> int:
    t = t.replace(tzinfo=timezone.utc)
    return int((t.hour * 3_600 + t.minute * 60 + t.second) * 1e9 + t.microsecond * 1e3)


def _date_to_pl_date(d: date) -> int:
    dt = datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) // (3600 * 24)


def _timedelta_to_pl_timedelta(td: timedelta, time_unit: TimeUnit | None = None) -> int:
    if time_unit == "ns":
        return int(td.total_seconds() * 1e9)
    elif time_unit == "us":
        return int(td.total_seconds() * 1e6)
    elif time_unit == "ms":
        return int(td.total_seconds() * 1e3)
    elif time_unit is None:
        # python has us precision
        return int(td.total_seconds() * 1e6)
    else:
        raise ValueError(
            f"time_unit must be one of {{'ns', 'us', 'ms'}}, got {time_unit}"
        )


def _to_python_time(value: int) -> time:
    """Convert polars int64 (ns) timestamp to python time object."""
    if value == 0:
        return time(microsecond=0)
    else:
        seconds, nanoseconds = divmod(value, 1_000_000_000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return time(
            hour=hours, minute=minutes, second=seconds, microsecond=nanoseconds // 1000
        )


def _to_python_timedelta(value: int | float, time_unit: TimeUnit = "ns") -> timedelta:
    if time_unit == "ns":
        return timedelta(microseconds=value // 1e3)
    elif time_unit == "us":
        return timedelta(microseconds=value)
    elif time_unit == "ms":
        return timedelta(milliseconds=value)
    else:
        raise ValueError(
            f"time_unit must be one of {{'ns', 'us', 'ms'}}, got {time_unit}"
        )


EPOCH = datetime(1970, 1, 1).replace(tzinfo=None)
EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

_fromtimestamp = datetime.fromtimestamp


@lru_cache(256)
def _to_python_date(value: int | float) -> date:
    """Convert polars int64 timestamp to Python date."""
    return (EPOCH_UTC + timedelta(seconds=value * 86400)).date()


def _to_python_datetime(
    value: int | float,
    time_unit: TimeUnit | None = "ns",
    time_zone: str | None = None,
) -> datetime:
    """Convert polars int64 timestamp to Python datetime."""
    if not time_zone:
        if time_unit == "us":
            return EPOCH + timedelta(microseconds=value)
        elif time_unit == "ns":
            return EPOCH + timedelta(microseconds=value // 1000)
        elif time_unit == "ms":
            return EPOCH + timedelta(milliseconds=value)
        else:
            raise ValueError(
                f"time_unit must be one of {{'ns','us','ms'}}, got {time_unit}"
            )
    elif _ZONEINFO_AVAILABLE:
        if time_unit == "us":
            dt = EPOCH_UTC + timedelta(microseconds=value)
        elif time_unit == "ns":
            dt = EPOCH_UTC + timedelta(microseconds=value // 1000)
        elif time_unit == "ms":
            dt = EPOCH_UTC + timedelta(milliseconds=value)
        else:
            raise ValueError(
                f"time_unit must be one of {{'ns','us','ms'}}, got {time_unit}"
            )
        return _localize(dt, time_zone)
    else:
        raise ImportError(
            "Install polars[timezone] to handle datetimes with timezones."
        )


def _localize(dt: datetime, time_zone: str) -> datetime:
    # zone info installation should already be checked
    _tzinfo: ZoneInfo | tzinfo
    try:
        _tzinfo = get_zoneinfo(time_zone)
    except zoneinfo.ZoneInfoNotFoundError:
        # try fixed offset, which is not supported by ZoneInfo
        _tzinfo = _parse_fixed_tz_offset(time_zone)

    return dt.astimezone(_tzinfo)


def _datetime_for_anyvalue(dt: datetime) -> tuple[float, int]:
    """Used in pyo3 anyvalue conversion."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # returns (s, ms)
    return (dt.replace(microsecond=0).timestamp(), dt.microsecond)


def _datetime_for_anyvalue_windows(dt: datetime) -> tuple[float, int]:
    """Used in pyo3 anyvalue conversion."""
    if dt.tzinfo is None:
        dt = _localize(dt, "UTC")
    # returns (s, ms)
    return (dt.replace(microsecond=0).timestamp(), dt.microsecond)


# cache here as we have a single tz per column
# and this function will be called on every conversion
@lru_cache(16)
def _parse_fixed_tz_offset(offset: str) -> tzinfo:
    try:
        # use fromisoformat to parse the offset
        dt_offset = datetime.fromisoformat("2000-01-01T00:00:00" + offset)

        # alternatively, we parse the offset ourselves extracting hours and
        # minutes, then we can construct:
        # tzinfo=timezone(timedelta(hours=..., minutes=...))
    except ValueError:
        raise ValueError(f"Offset: {offset} not understood.") from None

    return dt_offset.tzinfo  # type: ignore[return-value]


def _to_python_decimal(
    sign: int, digits: Sequence[int], prec: int, scale: int
) -> Decimal:
    return _create_decimal_with_prec(prec)((sign, digits, scale))


@lru_cache(None)
def _create_decimal_with_prec(
    precision: int,
) -> Callable[[tuple[int, Sequence[int], int]], Decimal]:
    # pre-cache contexts so we don't have to spend time on recreating them every time
    return Context(prec=precision).create_decimal
