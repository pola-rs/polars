"""Utility functions."""
from __future__ import annotations

import functools
import inspect
import os
import sys
import warnings
from collections.abc import MappingView, Reversible, Sized
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Sequence,
    TypeVar,
    overload,
)

import polars.internals as pli
from polars.datatypes import (
    Date,
    Datetime,
    Int64,
    PolarsDataType,
    is_polars_dtype,
)
from polars.dependencies import _ZONEINFO_AVAILABLE, zoneinfo

try:
    from polars.polars import PyExpr
    from polars.polars import pool_size as _pool_size

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

# This code block is due to a typing issue with backports.zoneinfo package:
# https://github.com/pganssle/zoneinfo/issues/125
if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
elif _ZONEINFO_AVAILABLE:
    from backports.zoneinfo._zoneinfo import ZoneInfo

if sys.version_info >= (3, 10):
    from typing import ParamSpec, TypeGuard
else:
    from typing_extensions import ParamSpec, TypeGuard

# note: reversed views don't match as instances of MappingView
if sys.version_info >= (3, 11):
    _views: list[Reversible[Any]] = [{}.keys(), {}.values(), {}.items()]
    _reverse_mapping_views = tuple(type(reversed(view)) for view in _views)

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


def _is_generator(val: object) -> bool:
    return (
        (isinstance(val, (Generator, Iterable)) and not isinstance(val, Sized))
        or isinstance(val, MappingView)
        or (sys.version_info >= (3, 11) and isinstance(val, _reverse_mapping_views))
    )


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


def range_to_series(
    name: str, rng: range, dtype: PolarsDataType | None = Int64
) -> pli.Series:
    """Fast conversion of the given range to a Series."""
    return pli.arange(
        low=rng.start,
        high=rng.stop,
        step=rng.step,
        eager=True,
        dtype=dtype,
    ).rename(name, in_place=True)


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
                "'columns' arg should contain a list of all integers or all strings"
                " values."
            )
        if columns and len(set(columns)) != len(columns):
            raise ValueError(
                f"'columns' arg should only have unique values. Got '{columns}'."
            )
        if projection and len(set(projection)) != len(projection):
            raise ValueError(
                f"'columns' arg should only have unique values. Got '{projection}'."
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

            utc = ZoneInfo("UTC")
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
        tzinfo = ZoneInfo(tz)
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


def arrlen(obj: Any) -> int | None:
    """Return length of (non-string) sequence object; returns None for non-sequences."""
    try:
        return None if isinstance(obj, str) else len(obj)
    except TypeError:
        return None


def normalise_filepath(path: str | Path, check_not_directory: bool = True) -> str:
    """Create a string path, expanding the home directory if present."""
    path = os.path.expanduser(path)
    if check_not_directory and os.path.exists(path) and os.path.isdir(path):
        raise IsADirectoryError(f"Expected a file path; {path!r} is a directory")
    return path


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


def redirect(from_to: dict[str, str]) -> Callable[[type[T]], type[T]]:
    """
    Class decorator allowing deprecation/transition from one method name to another.

    The parameters must be the same (unless they are being renamed, in
    which case you can use this in conjunction with @deprecated_alias).
    """

    def _redirecting_getattr_(obj: T, item: Any) -> Any:
        if isinstance(item, str) and item in from_to:
            new_item = from_to[item]
            warnings.warn(
                f"`{type(obj).__name__}.{item}` has been renamed; this"
                f" redirect is temporary, please use `.{new_item}` instead",
                category=DeprecationWarning,
                stacklevel=2,
            )
            item = new_item
        return obj.__getattribute__(item)

    def _cls_(cls: type[T]) -> type[T]:
        # note: __getattr__ is only invoked if item isn't found on the class
        cls.__getattr__ = _redirecting_getattr_  # type: ignore[attr-defined]
        return cls

    return _cls_


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


def deprecate_nonkeyword_arguments(
    allowed_args: list[str] | None = None,
    message: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to deprecate the use of non-keyword arguments of a function.

    Parameters
    ----------
    allowed_args
        The names of some first arguments of the decorated function that are allowed to
        be given as positional arguments. Should include "self" when decorating class
        methods. If set to None (default), equal to all arguments that do not have a
        default value.
    message
        Optionally overwrite the default warning message.
    """

    def decorate(fn: Callable[P, T]) -> Callable[P, T]:
        old_sig = inspect.signature(fn)

        if allowed_args is not None:
            allow_args = allowed_args
        else:
            allow_args = [
                p.name
                for p in old_sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]

        new_params = [
            p.replace(kind=p.KEYWORD_ONLY)
            if (
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.name not in allow_args
            )
            else p
            for p in old_sig.parameters.values()
        ]
        new_params.sort(key=lambda p: p.kind)

        new_sig = old_sig.replace(parameters=new_params)

        num_allowed_args = len(allow_args)
        if message is None:
            msg_format = (
                f"All arguments of {fn.__qualname__}{{except_args}} will be keyword-only in the next breaking release."
                " Use keyword arguments to silence this warning."
            )
            msg = msg_format.format(except_args=_format_argument_list(allow_args))
        else:
            msg = message

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if len(args) > num_allowed_args:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        return wrapper

    return decorate


def _format_argument_list(allowed_args: list[str]) -> str:
    """
    Format allowed arguments list for use in the warning message of
    ``deprecate_nonkeyword_arguments``.
    """  # noqa: D205
    if "self" in allowed_args:
        allowed_args.remove("self")
    if not allowed_args:
        return ""
    elif len(allowed_args) == 1:
        return f" except for {allowed_args[0]!r}"
    else:
        last = allowed_args[-1]
        args = ", ".join([f"{x!r}" for x in allowed_args[:-1]])
        return f" except for {args} and {last!r}"


# when building docs (with Sphinx) we need access to the functions
# associated with the namespaces from the class, as we don't have
# an instance; @sphinx_accessor is a @property that allows this.
NS = TypeVar("NS")


class sphinx_accessor(property):
    def __get__(  # type: ignore[override]
        self,
        instance: Any,
        cls: type[NS],
    ) -> NS:
        try:
            return self.fget(  # type: ignore[misc]
                instance if isinstance(instance, cls) else cls
            )
        except AttributeError:
            return None  # type: ignore[return-value]


def scale_bytes(sz: int, unit: SizeUnit) -> int | float:
    """Scale size in bytes to other size units (eg: "kb", "mb", "gb", "tb")."""
    if unit in {"b", "bytes"}:
        return sz
    elif unit in {"kb", "kilobytes"}:
        return sz / 1024
    elif unit in {"mb", "megabytes"}:
        return sz / 1024**2
    elif unit in {"gb", "gigabytes"}:
        return sz / 1024**3
    elif unit in {"tb", "terabytes"}:
        return sz / 1024**4
    else:
        raise ValueError(
            f"unit must be one of {{'b', 'kb', 'mb', 'gb', 'tb'}}, got {unit!r}"
        )
