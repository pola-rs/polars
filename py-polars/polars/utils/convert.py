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

    def get_zoneinfo(key: str) -> ZoneInfo:  # noqa: D103
        pass

else:

    @lru_cache(None)
    def get_zoneinfo(key: str) -> ZoneInfo:  # noqa: D103
        return zoneinfo.ZoneInfo(key)


# note: reversed views don't match as instances of MappingView
if sys.version_info >= (3, 11):
    _views: list[Reversible[Any]] = [{}.keys(), {}.values(), {}.items()]
    _reverse_mapping_views = tuple(type(reversed(view)) for view in _views)

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
NS_PER_SECOND = 1_000_000_000
US_PER_SECOND = 1_000_000
MS_PER_SECOND = 1_000

EPOCH = datetime(1970, 1, 1).replace(tzinfo=None)
EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _timestamp_in_seconds(dt: datetime) -> int:
    du = dt - EPOCH_UTC
    return du.days * SECONDS_PER_DAY + du.seconds


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
            corrected_seconds = SECONDS_PER_DAY - (td.seconds + (td.microseconds > 0))
            s = corrected_seconds and f"{corrected_seconds}s" or ""
            us = td.microseconds and f"{10**6 - td.microseconds}us" or ""

    return f"{d}{s}{us}"


def _negate_duration(duration: str) -> str:
    if duration.startswith("-"):
        return duration[1:]
    return f"-{duration}"


def _time_to_pl_time(t: time) -> int:
    t = t.replace(tzinfo=timezone.utc)
    seconds = t.hour * SECONDS_PER_HOUR + t.minute * 60 + t.second
    microseconds = t.microsecond
    return seconds * NS_PER_SECOND + microseconds * 1_000


def _date_to_pl_date(d: date) -> int:
    dt = datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) // SECONDS_PER_DAY


def _datetime_to_pl_timestamp(dt: datetime, time_unit: TimeUnit | None) -> int:
    """Convert a python datetime to a timestamp in given time unit."""
    if dt.tzinfo is None:
        # Make sure to use UTC rather than system time zone.
        dt = dt.replace(tzinfo=timezone.utc)
    microseconds = dt.microsecond
    seconds = _timestamp_in_seconds(dt)
    if time_unit == "ns":
        return seconds * NS_PER_SECOND + microseconds * 1_000
    elif time_unit == "us" or time_unit is None:
        return seconds * US_PER_SECOND + microseconds
    elif time_unit == "ms":
        return seconds * MS_PER_SECOND + microseconds // 1_000
    raise ValueError(
        f"`time_unit` must be one of {{'ms', 'us', 'ns'}}, got {time_unit!r}"
    )


def _timedelta_to_pl_timedelta(td: timedelta, time_unit: TimeUnit | None) -> int:
    """Convert a Python timedelta object to a total number of subseconds."""
    microseconds = td.microseconds
    seconds = td.days * SECONDS_PER_DAY + td.seconds
    if time_unit == "ns":
        return seconds * NS_PER_SECOND + microseconds * 1_000
    elif time_unit == "us" or time_unit is None:
        return seconds * US_PER_SECOND + microseconds
    elif time_unit == "ms":
        return seconds * MS_PER_SECOND + microseconds // 1_000


def _to_python_time(value: int) -> time:
    """Convert polars int64 (ns) timestamp to python time object."""
    if value == 0:
        return time(microsecond=0)
    else:
        seconds, nanoseconds = divmod(value, NS_PER_SECOND)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return time(
            hour=hours, minute=minutes, second=seconds, microsecond=nanoseconds // 1_000
        )


def _to_python_timedelta(
    value: int | float, time_unit: TimeUnit | None = "ns"
) -> timedelta:
    if time_unit == "ns":
        return timedelta(microseconds=value // 1_000)
    elif time_unit == "us":
        return timedelta(microseconds=value)
    elif time_unit == "ms":
        return timedelta(milliseconds=value)
    else:
        raise ValueError(
            f"`time_unit` must be one of {{'ns', 'us', 'ms'}}, got {time_unit!r}"
        )


@lru_cache(256)
def _to_python_date(value: int | float) -> date:
    """Convert polars int64 timestamp to Python date."""
    return (EPOCH_UTC + timedelta(seconds=value * SECONDS_PER_DAY)).date()


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
            return EPOCH + timedelta(microseconds=value // 1_000)
        elif time_unit == "ms":
            return EPOCH + timedelta(milliseconds=value)
        else:
            raise ValueError(
                f"`time_unit` must be one of {{'ns', 'us', 'ms'}}, got {time_unit!r}"
            )
    elif _ZONEINFO_AVAILABLE:
        if time_unit == "us":
            dt = EPOCH_UTC + timedelta(microseconds=value)
        elif time_unit == "ns":
            dt = EPOCH_UTC + timedelta(microseconds=value // 1_000)
        elif time_unit == "ms":
            dt = EPOCH_UTC + timedelta(milliseconds=value)
        else:
            raise ValueError(
                f"`time_unit` must be one of {{'ns', 'us', 'ms'}}, got {time_unit!r}"
            )
        return _localize(dt, time_zone)
    else:
        raise ImportError(
            "install polars[timezone] to handle datetimes with time zone information"
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


def _datetime_for_anyvalue(dt: datetime) -> tuple[int, int]:
    """Used in pyo3 anyvalue conversion."""
    # returns (s, ms)
    if dt.tzinfo is None:
        return (
            _timestamp_in_seconds(dt.replace(tzinfo=timezone.utc)),
            dt.microsecond,
        )
    return (_timestamp_in_seconds(dt), dt.microsecond)


def _datetime_for_anyvalue_windows(dt: datetime) -> tuple[float, int]:
    """Used in pyo3 anyvalue conversion."""
    if dt.tzinfo is None:
        dt = _localize(dt, "UTC")
    # returns (s, ms)
    return (_timestamp_in_seconds(dt), dt.microsecond)


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
        raise ValueError(f"offset: {offset!r} not understood") from None

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
