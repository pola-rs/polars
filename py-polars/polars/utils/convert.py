from __future__ import annotations

import functools
import sys
from datetime import datetime, time, timedelta, timezone
from decimal import Context
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, overload

from polars.datatypes import Date, Datetime
from polars.dependencies import _ZONEINFO_AVAILABLE, zoneinfo

if TYPE_CHECKING:
    from collections.abc import Reversible
    from datetime import date, tzinfo
    from decimal import Decimal

    from polars.type_aliases import PolarsDataType, TimeUnit

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
    from functools import lru_cache

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


def _datetime_to_pl_timestamp(dt: datetime, tu: TimeUnit | None) -> int:
    """Convert a python datetime to a timestamp in nanoseconds."""
    dt = dt.replace(tzinfo=timezone.utc)
    if tu == "ns":
        nanos = dt.microsecond * 1000
        return int(dt.timestamp()) * 1_000_000_000 + nanos
    elif tu == "us":
        micros = dt.microsecond
        return int(dt.timestamp()) * 1_000_000 + micros
    elif tu == "ms":
        millis = dt.microsecond // 1000
        return int(dt.timestamp()) * 1_000 + millis
    elif tu is None:
        # python has us precision
        micros = dt.microsecond
        return int(dt.timestamp()) * 1_000_000 + micros
    else:
        raise ValueError(f"tu must be one of {{'ns', 'us', 'ms'}}, got {tu}")


def _time_to_pl_time(t: time) -> int:
    t = t.replace(tzinfo=timezone.utc)
    return int((t.hour * 3_600 + t.minute * 60 + t.second) * 1e9 + t.microsecond * 1e3)


def _date_to_pl_date(d: date) -> int:
    dt = datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) // (3600 * 24)


def _timedelta_to_pl_timedelta(td: timedelta, tu: TimeUnit | None = None) -> int:
    if tu == "ns":
        return int(td.total_seconds() * 1e9)
    elif tu == "us":
        return int(td.total_seconds() * 1e6)
    elif tu == "ms":
        return int(td.total_seconds() * 1e3)
    elif tu is None:
        # python has us precision
        return int(td.total_seconds() * 1e6)
    else:
        raise ValueError(f"tu must be one of {{'ns', 'us', 'ms'}}, got {tu}")


def _to_python_time(value: int) -> time:
    if value == 0:
        return time(microsecond=0)
    value = value // 1_000
    microsecond = value
    seconds = (microsecond // 1000_000) % 60
    minutes = (microsecond // (1000_000 * 60)) % 60
    hours = (microsecond // (1000_000 * 60 * 60)) % 24
    microsecond = microsecond - (seconds + minutes * 60 + hours * 3600) * 1000_000

    return time(hour=hours, minute=minutes, second=seconds, microsecond=microsecond)


def _to_python_timedelta(value: int | float, tu: TimeUnit = "ns") -> timedelta:
    if tu == "ns":
        return timedelta(microseconds=value // 1e3)
    elif tu == "us":
        return timedelta(microseconds=value)
    elif tu == "ms":
        return timedelta(milliseconds=value)
    else:
        raise ValueError(f"tu must be one of {{'ns', 'us', 'ms'}}, got {tu}")


EPOCH = datetime(1970, 1, 1).replace(tzinfo=None)


def _to_python_datetime(
    value: int | float,
    dtype: PolarsDataType,
    tu: TimeUnit | None = "ns",
    tz: str | None = None,
) -> date | datetime:
    if dtype == Date:
        # days to seconds
        # important to create from utc. Not doing this leads
        # to inconsistencies dependent on the timezone you are in.
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        dt += timedelta(seconds=value * 3600 * 24)
        return dt.date()
    elif dtype == Datetime:
        if tz is None or tz == "":
            if tu == "ns":
                # nanoseconds to seconds
                dt = EPOCH + timedelta(microseconds=value / 1000)
            elif tu == "us":
                dt = EPOCH + timedelta(microseconds=value)
            elif tu == "ms":
                # milliseconds to seconds
                dt = datetime.utcfromtimestamp(value / 1000)
            else:
                raise ValueError(f"tu must be one of {{'ns', 'us', 'ms'}}, got {tu}")
        else:
            if not _ZONEINFO_AVAILABLE:
                raise ImportError(
                    "Install polars[timezone] to handle datetimes with timezones."
                )

            utc = get_zoneinfo("UTC")
            if tu == "ns":
                # nanoseconds to seconds
                dt = datetime.fromtimestamp(0, tz=utc) + timedelta(
                    microseconds=value / 1000
                )
            elif tu == "us":
                dt = datetime.fromtimestamp(0, tz=utc) + timedelta(microseconds=value)
            elif tu == "ms":
                # milliseconds to seconds
                dt = datetime.fromtimestamp(value / 1000, tz=utc)
            else:
                raise ValueError(f"tu must be one of {{'ns', 'us', 'ms'}}, got {tu}")
            return _localize(dt, tz)

        return dt
    else:
        raise NotImplementedError  # pragma: no cover


def _localize(dt: datetime, tz: str) -> datetime:
    # zone info installation should already be checked
    _tzinfo: ZoneInfo | tzinfo
    try:
        _tzinfo = get_zoneinfo(tz)
    except zoneinfo.ZoneInfoNotFoundError:
        # try fixed offset, which is not supported by ZoneInfo
        _tzinfo = _parse_fixed_tz_offset(tz)

    return dt.astimezone(_tzinfo)


# cache here as we have a single tz per column
# and this function will be called on every conversion
@functools.lru_cache(16)
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


@functools.lru_cache(None)
def _create_decimal_with_prec(
    precision: int,
) -> Callable[[tuple[int, Sequence[int], int]], Decimal]:
    # pre-cache contexts so we don't have to spend time on recreating them every time
    return Context(prec=precision).create_decimal


def _tzinfo_to_str(tzinfo: tzinfo) -> str:
    if tzinfo == timezone.utc:
        return "UTC"
    if isinstance(tzinfo, timezone):
        return str(tzinfo).replace("UTC", "")
    return str(tzinfo)
