"""Utility functions."""
from __future__ import annotations

import functools
import os
import sys
import warnings
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, TypeVar, overload

import polars.internals as pli
from polars.datatypes import DataType, Date, Datetime, PolarsDataType, is_polars_dtype
from polars.dependencies import _ZONEINFO_AVAILABLE, zoneinfo

try:
    from polars.polars import PyExpr
    from polars.polars import pool_size as _pool_size

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

if sys.version_info >= (3, 10):
    from typing import ParamSpec, TypeGuard
else:
    from typing_extensions import ParamSpec, TypeGuard


if TYPE_CHECKING:
    from polars.internals.type_aliases import SizeUnit, TimeUnit


def _process_null_values(
    null_values: None | str | list[str] | dict[str, str] = None,
) -> None | str | list[str] | list[tuple[str, str]]:
    if isinstance(null_values, dict):
        return list(null_values.items())
    else:
        return null_values


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
        d = td.days and f"{td.days}d" or ""
        s = td.seconds and f"{td.seconds}s" or ""
        us = td.microseconds and f"{td.microseconds}us" or ""
        return f"{d}{s}{us}"


def _datetime_to_pl_timestamp(dt: datetime, tu: TimeUnit | None) -> int:
    """Convert a python datetime to a timestamp in nanoseconds."""
    if tu == "ns":
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e9)
    elif tu == "us":
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e6)
    elif tu == "ms":
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e3)
    elif tu is None:
        # python has us precision
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e6)
    else:
        raise ValueError(f"tu must be one of {{'ns', 'us', 'ms'}}, got {tu}")


def _time_to_pl_time(t: time) -> int:
    t = t.replace(tzinfo=timezone.utc)
    return int((t.hour * 3_600 + t.minute * 60 + t.second) * 1e9 + t.microsecond * 1e3)


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


def _date_to_pl_date(d: date) -> int:
    dt = datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) // (3600 * 24)


def _is_iterable_of(val: Iterable[object], eltype: type | tuple[type, ...]) -> bool:
    """Check whether the given iterable is of the given type(s)."""
    return all(isinstance(x, eltype) for x in val)


def is_bool_sequence(val: object) -> TypeGuard[Sequence[bool]]:
    """Check whether the given sequence is a sequence of booleans."""
    return isinstance(val, Sequence) and _is_iterable_of(val, bool)


def is_dtype_sequence(val: object) -> TypeGuard[Sequence[PolarsDataType]]:
    """Check whether the given object is a sequence of polars DataTypes."""
    return isinstance(val, Sequence) and all(is_polars_dtype(x) for x in val)


def is_int_sequence(val: object) -> TypeGuard[Sequence[int]]:
    """Check whether the given sequence is a sequence of integers."""
    return isinstance(val, Sequence) and _is_iterable_of(val, int)


def is_expr_sequence(val: object) -> TypeGuard[Sequence[pli.Expr]]:
    """Check whether the given object is a sequence of Exprs."""
    return isinstance(val, Sequence) and _is_iterable_of(val, pli.Expr)


def is_pyexpr_sequence(val: object) -> TypeGuard[Sequence[PyExpr]]:
    """Check whether the given object is a sequence of PyExprs."""
    return isinstance(val, Sequence) and _is_iterable_of(val, PyExpr)


def is_str_sequence(
    val: object, *, allow_str: bool = False
) -> TypeGuard[Sequence[str]]:
    """
    Check that `val` is a sequence of strings.

    Note that a single string is a sequence of strings by definition, use
    `allow_str=False` to return False on a single string.
    """
    if allow_str is False and isinstance(val, str):
        return False
    return isinstance(val, Sequence) and _is_iterable_of(val, str)


def range_to_slice(rng: range) -> slice:
    """Return the given range as an equivalent slice."""
    return slice(rng.start, rng.stop, rng.step)


def handle_projection_columns(
    columns: Sequence[str] | Sequence[int] | str | None,
) -> tuple[list[int] | None, list[str] | None]:
    """Disambiguates between columns specified as integers vs. strings."""
    projection: list[int] | None = None
    if columns:
        if isinstance(columns, str):
            columns = [columns]
        elif is_int_sequence(columns):
            projection = list(columns)
            columns = None
        elif not is_str_sequence(columns):
            raise ValueError(
                "columns arg should contain a list of all integers or all strings"
                " values."
            )
    return projection, columns  # type: ignore[return-value]


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


def _prepare_row_count_args(
    row_count_name: str | None = None,
    row_count_offset: int = 0,
) -> tuple[str, int] | None:
    if row_count_name is not None:
        return (row_count_name, row_count_offset)
    else:
        return None


EPOCH = datetime(1970, 1, 1).replace(tzinfo=None)


def _to_python_datetime(
    value: int | float,
    dtype: type[DataType],
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

            utc = zoneinfo.ZoneInfo("UTC")
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


def _localize(dt: datetime, tz: str) -> datetime:
    # zone info installation should already be checked
    try:
        tzinfo = zoneinfo.ZoneInfo(tz)
    except zoneinfo.ZoneInfoNotFoundError:
        # try fixed offset, which is not supported by ZoneInfo
        tzinfo = _parse_fixed_tz_offset(tz)  # type: ignore[assignment]

    return dt.astimezone(tzinfo)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def format_path(path: str | Path) -> str:
    """Create a string path, expanding the home directory if present."""
    return os.path.expanduser(path)


def threadpool_size() -> int:
    """Get the size of polars' thread pool."""
    return _pool_size()


P = ParamSpec("P")
T = TypeVar("T")


def deprecated_alias(**aliases: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Deprecate a function or method argument.

    Decorator for deprecated function and method arguments. Use as follows:

    @deprecated_alias(old_arg='new_arg')
    def myfunc(new_arg):
        ...
    """

    def deco(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            _rename_kwargs(fn.__name__, kwargs, aliases)
            return fn(*args, **kwargs)

        return wrapper

    return deco


def _rename_kwargs(
    func_name: str, kwargs: dict[str, object], aliases: dict[str, str]
) -> None:
    """
    Rename the keyword arguments of a function.

    Helper function for deprecating function and method arguments.
    """
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(
                    f"{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is deprecated, use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is deprecated as an argument to `{func_name}`; use"
                    f" `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)


if not os.getenv("BUILDING_SPHINX_DOCS"):
    # if NOT building docs we use a simple @property decorator for namespace accessors,
    # which plays much better with mypy, pylint, and the various IDEs' autocomplete.
    accessor = property
else:
    # however, when building docs (with Sphinx) we need access to the functions
    # associated with the namespaces from the class, as we don't have an instance.
    NS = TypeVar("NS")

    class accessor(property):  # type: ignore[no-redef]
        def __get__(self, instance: Any, cls: type[NS]) -> NS:  # type: ignore[override]
            return self.fget(  # type: ignore[misc]
                instance if isinstance(instance, cls) else cls
            )


def scale_bytes(sz: int, to: SizeUnit) -> int | float:
    """Scale size in bytes to other size units (eg: "kb", "mb", "gb", "tb")."""
    scaling_factor = {
        "b": 1,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
    }[to[0]]
    if scaling_factor > 1:
        return sz / scaling_factor
    return sz
