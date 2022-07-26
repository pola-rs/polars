"""Utility functions."""
from __future__ import annotations

import ctypes
import functools
import os
import sys
import warnings
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from polars.datatypes import DataType, Date, Datetime

try:
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
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


def _process_null_values(
    null_values: None | str | list[str] | dict[str, str] = None,
) -> None | str | list[str] | list[tuple[str, str]]:
    if isinstance(null_values, dict):
        return list(null_values.items())
    else:
        return null_values


# https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
def _ptr_to_numpy(ptr: int, len: int, ptr_type: Any) -> np.ndarray:
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


def _datetime_to_pl_timestamp(dt: datetime, tu: str | None) -> int:
    """Convert a python datetime to a timestamp in nanoseconds."""
    if tu == "ns":
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e9)
    elif tu == "us":
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e6)
    elif tu == "ms":
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e3)
    if tu is None:
        # python has us precision
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e6)
    else:
        raise ValueError("expected one of {'ns', 'us', 'ms'}")


def _timedelta_to_pl_timedelta(td: timedelta, tu: str | None = None) -> int:
    if tu == "ns":
        return int(td.total_seconds() * 1e9)
    elif tu == "us":
        return int(td.total_seconds() * 1e6)
    elif tu == "ms":
        return int(td.total_seconds() * 1e3)
    if tu is None:
        if timedelta_in_nanoseconds_window(td):
            return int(td.total_seconds() * 1e9)
        else:
            return int(td.total_seconds() * 1e3)
    else:
        raise ValueError("expected one of {'ns', 'us, 'ms'}")


def _date_to_pl_date(d: date) -> int:
    dt = datetime.combine(d, datetime.min.time()).replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) // (3600 * 24)


def _is_iterable_of(val: Iterable, itertype: type, eltype: type) -> bool:
    """Check whether the given iterable is of a certain type."""
    return isinstance(val, itertype) and all(isinstance(x, eltype) for x in val)


def is_bool_sequence(val: Sequence[object]) -> TypeGuard[Sequence[bool]]:
    """Check whether the given sequence is a sequence of booleans."""
    return _is_iterable_of(val, Sequence, bool)


def is_int_sequence(val: Sequence[object]) -> TypeGuard[Sequence[int]]:
    """Check whether the given sequence is a sequence of integers."""
    return _is_iterable_of(val, Sequence, int)


def is_str_sequence(
    val: Sequence[object], allow_str: bool = False
) -> TypeGuard[Sequence[str]]:
    """
    Check that `val` is a sequence of strings.

    Note that a single string is a sequence of strings by definition, use
    `allow_str=False` to return False on a single string.
    """
    if (not allow_str) and isinstance(val, str):
        return False
    return _is_iterable_of(val, Sequence, str)


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


def _to_python_timedelta(value: int | float, tu: str | None = "ns") -> timedelta:
    if tu == "ns":
        return timedelta(microseconds=value // 1e3)
    elif tu == "us":
        return timedelta(microseconds=value)
    elif tu == "ms":
        return timedelta(milliseconds=value)
    else:
        raise ValueError(f"time unit: {tu} not expected")


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
    tu: str | None = "ns",
    tz: str | None = None,
) -> date | datetime:
    if dtype == Date:
        # days to seconds
        # important to create from utc. Not doing this leads
        # to inconsistencies dependent on the timezone you are in.
        return datetime.utcfromtimestamp(value * 3600 * 24).date()
    elif dtype == Datetime:
        if tu == "ns":
            # nanoseconds to seconds
            dt = EPOCH + timedelta(microseconds=value / 1000)
        elif tu == "us":
            dt = EPOCH + timedelta(microseconds=value)
        elif tu == "ms":
            # milliseconds to seconds
            dt = datetime.utcfromtimestamp(value / 1_000)
        else:
            raise ValueError(f"time unit: {tu} not expected")
        if tz is not None and len(tz) > 0:
            import pytz

            timezone = pytz.timezone(tz)
            return timezone.localize(dt)
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


def deprecated_alias(**aliases: str) -> Callable:
    """
    Deprecate a function or method argument.

    Decorator for deprecated function and method arguments. Use as follows:

    @deprecated_alias(old_arg='new_arg')
    def myfunc(new_arg):
        ...
    """

    def deco(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            _rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def _rename_kwargs(
    func_name: str, kwargs: dict[str, str], aliases: dict[str, str]
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
