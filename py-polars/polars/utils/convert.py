from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta, timezone
from decimal import Context
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, overload

from polars.dependencies import _ZONEINFO_AVAILABLE, zoneinfo

if TYPE_CHECKING:
    from collections.abc import Reversible
    from datetime import tzinfo
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

EPOCH_DATE = date(1970, 1, 1)
EPOCH = datetime(1970, 1, 1).replace(tzinfo=None)
EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


@overload
def parse_as_duration_string(td: None) -> None:
    ...


@overload
def parse_as_duration_string(td: timedelta | str) -> str:
    ...


def parse_as_duration_string(td: timedelta | str | None) -> str | None:
    """Parse duration input as a Polars duration string."""
    if td is None or isinstance(td, str):
        return td
    return _timedelta_to_duration_string(td)


def _timedelta_to_duration_string(td: timedelta) -> str:
    """Convert a Python timedelta object to a Polars duration string."""
    # Positive duration
    if td.days >= 0:
        d = f"{td.days}d" if td.days != 0 else ""
        s = f"{td.seconds}s" if td.seconds != 0 else ""
        us = f"{td.microseconds}us" if td.microseconds != 0 else ""
    # Negative, whole days
    elif td.seconds == 0 and td.microseconds == 0:
        return f"{td.days}d"
    # Negative, other
    else:
        corrected_d = td.days + 1
        corrected_seconds = SECONDS_PER_DAY - (td.seconds + (td.microseconds > 0))
        d = f"{corrected_d}d" if corrected_d != 0 else "-"
        s = f"{corrected_seconds}s" if corrected_seconds != 0 else ""
        us = f"{10**6 - td.microseconds}us" if td.microseconds != 0 else ""

    return f"{d}{s}{us}"


def negate_duration_string(duration: str) -> str:
    """Negate a Polars duration string."""
    if duration.startswith("-"):
        return duration[1:]
    else:
        return f"-{duration}"


def date_to_int(d: date) -> int:
    """Convert a Python time object to an integer."""
    return (d - EPOCH_DATE).days


def time_to_int(t: time) -> int:
    """Convert a Python time object to an integer."""
    t = t.replace(tzinfo=timezone.utc)
    seconds = t.hour * SECONDS_PER_HOUR + t.minute * 60 + t.second
    microseconds = t.microsecond
    return seconds * NS_PER_SECOND + microseconds * 1_000


def _datetime_to_pl_timestamp(dt: datetime, time_unit: TimeUnit) -> int:
    """Convert a Python datetime object to an integer."""
    # Make sure to use UTC rather than system time zone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    seconds = _timestamp_in_seconds(dt)
    microseconds = dt.microsecond

    if time_unit == "us":
        return seconds * US_PER_SECOND + microseconds
    elif time_unit == "ns":
        return seconds * NS_PER_SECOND + microseconds * 1_000
    elif time_unit == "ms":
        return seconds * MS_PER_SECOND + microseconds // 1_000
    else:
        msg = f"`time_unit` must be one of {{'ms', 'us', 'ns'}}, got {time_unit!r}"
        raise ValueError(msg)


def _timestamp_in_seconds(dt: datetime) -> int:
    td = dt - EPOCH_UTC
    return td.days * SECONDS_PER_DAY + td.seconds


def _timedelta_to_pl_timedelta(td: timedelta, time_unit: TimeUnit) -> int:
    """Convert a Python timedelta object to an integer."""
    seconds = td.days * SECONDS_PER_DAY + td.seconds
    microseconds = td.microseconds

    if time_unit == "us":
        return seconds * US_PER_SECOND + microseconds
    elif time_unit == "ns":
        return seconds * NS_PER_SECOND + microseconds * 1_000
    elif time_unit == "ms":
        return seconds * MS_PER_SECOND + microseconds // 1_000
    else:
        msg = f"`time_unit` must be one of {{'ms', 'us', 'ns'}}, got {time_unit!r}"
        raise ValueError(msg)


@lru_cache(256)
def _to_python_date(value: int | float) -> date:
    """Convert an integer or float to a Python date object."""
    return EPOCH_DATE + timedelta(days=value)


def _to_python_time(value: int) -> time:
    """Convert an integer to a Python time object."""
    # Fast path for 00:00
    if value == 0:
        return time()

    seconds, nanoseconds = divmod(value, NS_PER_SECOND)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return time(
        hour=hours, minute=minutes, second=seconds, microsecond=nanoseconds // 1_000
    )


def _to_python_datetime(
    value: int | float,
    time_unit: TimeUnit,
    time_zone: str | None = None,
) -> datetime:
    """Convert an integer or float to a Python datetime object."""
    if time_unit == "us":
        td = timedelta(microseconds=value)
    elif time_unit == "ns":
        td = timedelta(microseconds=value // 1_000)
    elif time_unit == "ms":
        td = timedelta(milliseconds=value)
    else:
        msg = f"`time_unit` must be one of {{'ns', 'us', 'ms'}}, got {time_unit!r}"
        raise ValueError(msg)

    if time_zone is None:
        return EPOCH + td
    elif _ZONEINFO_AVAILABLE:
        dt = EPOCH_UTC + td
        return _localize(dt, time_zone)
    else:
        msg = "install polars[timezone] to handle datetimes with time zone information"
        raise ImportError(msg)


def _localize(dt: datetime, time_zone: str) -> datetime:
    # zone info installation should already be checked
    _tzinfo: ZoneInfo | tzinfo
    try:
        _tzinfo = get_zoneinfo(time_zone)
    except zoneinfo.ZoneInfoNotFoundError:
        # try fixed offset, which is not supported by ZoneInfo
        _tzinfo = _parse_fixed_tz_offset(time_zone)

    return dt.astimezone(_tzinfo)


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
        msg = f"offset: {offset!r} not understood"
        raise ValueError(msg) from None

    return dt_offset.tzinfo  # type: ignore[return-value]


def _to_python_timedelta(value: int | float, time_unit: TimeUnit) -> timedelta:
    """Convert an integer or float to a Python timedelta object."""
    if time_unit == "us":
        return timedelta(microseconds=value)
    elif time_unit == "ns":
        return timedelta(microseconds=value // 1_000)
    elif time_unit == "ms":
        return timedelta(milliseconds=value)
    else:
        msg = f"`time_unit` must be one of {{'ns', 'us', 'ms'}}, got {time_unit!r}"
        raise ValueError(msg)


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


def _datetime_for_any_value(dt: datetime) -> tuple[int, int]:
    """Used in PyO3 AnyValue conversion."""
    # returns (s, ms)
    if dt.tzinfo is None:
        return (
            _timestamp_in_seconds(dt.replace(tzinfo=timezone.utc)),
            dt.microsecond,
        )
    return (_timestamp_in_seconds(dt), dt.microsecond)


def _datetime_for_any_value_windows(dt: datetime) -> tuple[float, int]:
    """Used in PyO3 AnyValue conversion."""
    if dt.tzinfo is None:
        dt = _localize(dt, "UTC")
    # returns (s, ms)
    return (_timestamp_in_seconds(dt), dt.microsecond)
