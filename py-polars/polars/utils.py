"""Utility functions."""
from __future__ import annotations

import ctypes
import functools
import os
import sys
import warnings
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, TypeVar

import polars.internals as pli
from polars.datatypes import DataType, Date, Datetime

try:
    from polars.polars import PyExpr
    from polars.polars import pool_size as _pool_size

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

if sys.version_info >= (3, 10):
    from typing import ParamSpec, TypeGuard
else:
    from typing_extensions import ParamSpec, TypeGuard

if TYPE_CHECKING:
    from polars.internals.type_aliases import TimeUnit


def _process_null_values(
    null_values: None | str | list[str] | dict[str, str] = None,
) -> None | str | list[str] | list[tuple[str, str]]:
    if isinstance(null_values, dict):
        return list(null_values.items())
    else:
        return null_values


# https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
def _ptr_to_numpy(ptr: int, len: int, ptr_type: Any) -> np.ndarray[Any, Any]:
    """
    Create a memory block view as a numpy array.

    Parameters
    ----------
    ptr
        C/Rust ptr casted to usize.
    len
        Length of the array values.
    ptr_type
        Example:
            f32: ctypes.c_float)

    Returns
    -------
    View of memory block as numpy array.

    """
    if not _NUMPY_AVAILABLE:
        raise ImportError("'numpy' is required for this functionality.")
    ptr_ctype = ctypes.cast(ptr, ctypes.POINTER(ptr_type))
    return np.ctypeslib.as_array(ptr_ctype, (len,))


def _timedelta_to_pl_duration(td: timedelta) -> str:
    return f"{td.days}d{td.seconds}s{td.microseconds}us"


def in_nanoseconds_window(dt: datetime) -> bool:
    """Check whether the given datetime can be represented as a Unix timestamp."""
    return 1386 < dt.year < 2554


def timedelta_in_nanoseconds_window(td: timedelta) -> bool:
    """Check whether the given timedelta can be represented as a Unix timestamp."""
    return in_nanoseconds_window(datetime(1970, 1, 1) + td)


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


def _is_iterable_of(val: Iterable[object], eltype: type) -> bool:
    """Check whether the given iterable is of a certain type."""
    return all(isinstance(x, eltype) for x in val)


def is_bool_sequence(val: object) -> TypeGuard[Sequence[bool]]:
    """Check whether the given sequence is a sequence of booleans."""
    if isinstance(val, Sequence):
        return _is_iterable_of(val, bool)
    else:
        return False


def is_int_sequence(val: object) -> TypeGuard[Sequence[int]]:
    """Check whether the given sequence is a sequence of integers."""
    if isinstance(val, Sequence):
        return _is_iterable_of(val, int)
    else:
        return False


def is_expr_sequence(val: object) -> TypeGuard[Sequence[pli.Expr]]:
    """Check whether the given object is a sequence of Exprs."""
    if isinstance(val, Sequence):
        return _is_iterable_of(val, pli.Expr)
    else:
        return False


def is_pyexpr_sequence(val: object) -> TypeGuard[Sequence[PyExpr]]:
    """Check whether the given object is a sequence of PyExprs."""
    if isinstance(val, Sequence):
        return _is_iterable_of(val, PyExpr)
    else:
        return False


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
    if isinstance(val, Sequence):
        return _is_iterable_of(val, str)
    else:
        return False


def range_to_slice(rng: range) -> slice:
    """Return the given range as an equivalent slice."""
    return slice(rng.start, rng.stop, rng.step)


def handle_projection_columns(
    columns: list[str] | list[int] | None,
) -> tuple[list[int] | None, list[str] | None]:
    """Disambiguates between columns specified as integers vs. strings."""
    projection: list[int] | None = None
    if columns:
        if is_int_sequence(columns):
            projection = columns  # type: ignore[assignment]
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
        if tz is not None and len(tz) > 0:
            try:
                import pytz
            except ImportError:
                raise ImportError(
                    "pytz is not installed. Please run `pip install pytz`."
                ) from None

            return pytz.timezone(tz).localize(dt)
        return dt
    else:
        raise NotImplementedError  # pragma: no cover


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
    """Get the size of polars; thread pool."""
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
