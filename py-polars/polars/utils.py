import ctypes
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from polars.datatypes import DataType, Date, Datetime

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard  # pragma: no cover


def _process_null_values(
    null_values: Union[None, str, List[str], Dict[str, str]] = None,
) -> Union[None, str, List[str], List[Tuple[str, str]]]:
    if isinstance(null_values, dict):
        return list(null_values.items())
    else:
        return null_values


# https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
def _ptr_to_numpy(ptr: int, len: int, ptr_type: Any) -> np.ndarray:
    """

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
    ptr_ctype = ctypes.cast(ptr, ctypes.POINTER(ptr_type))
    return np.ctypeslib.as_array(ptr_ctype, (len,))


def _timedelta_to_pl_duration(td: timedelta) -> str:
    return f"{td.days}d{td.seconds}s{td.microseconds}us"


def in_nanoseconds_window(dt: datetime) -> bool:
    return 1386 < dt.year < 2554


def timedelta_in_nanoseconds_window(td: timedelta) -> bool:
    return in_nanoseconds_window(datetime(1970, 1, 1) + td)


def _datetime_to_pl_timestamp(dt: datetime, tu: Optional[str]) -> int:
    """
    Converts a python datetime to a timestamp in nanoseconds
    """
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
        raise ValueError("expected on of {'ns', 'ms'}")


def _timedelta_to_pl_timedelta(td: timedelta, tu: Optional[str] = None) -> int:
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


def is_str_sequence(
    val: Sequence[object], allow_str: bool = False
) -> TypeGuard[Sequence[str]]:
    """
    Checks that `val` is a sequence of strings. Note that a single string is a sequence of strings
    by definition, use `allow_str=False` to return False on a single string
    """
    if (not allow_str) and isinstance(val, str):
        return False
    return _is_iterable_of(val, Sequence, str)


def is_int_sequence(val: Sequence[object]) -> TypeGuard[Sequence[int]]:
    return _is_iterable_of(val, Sequence, int)


def _is_iterable_of(val: Iterable, itertype: Type, eltype: Type) -> bool:
    return isinstance(val, itertype) and all(isinstance(x, eltype) for x in val)


def range_to_slice(rng: range) -> slice:
    step: Optional[int]
    # maybe we can slice instead of take by indices
    if rng.step != 1:
        step = rng.step
    else:
        step = None
    return slice(rng.start, rng.stop, step)


def handle_projection_columns(
    columns: Optional[Union[List[str], List[int]]]
) -> Tuple[Optional[List[int]], Optional[List[str]]]:
    projection: Optional[List[int]] = None
    if columns:
        if is_int_sequence(columns):
            projection = columns  # type: ignore
            columns = None
        elif not is_str_sequence(columns):
            raise ValueError(
                "columns arg should contain a list of all integers or all strings values."
            )
    return projection, columns  # type: ignore


def _to_python_time(value: int) -> time:
    value = value // 1_000
    microsecond = value
    seconds = (microsecond // 1000_000) % 60
    minutes = (microsecond // (1000_000 * 60)) % 60
    hours = (microsecond // (1000_000 * 60 * 60)) % 24

    microsecond = microsecond % seconds * 1000_000

    return time(hour=hours, minute=minutes, second=seconds, microsecond=microsecond)


def _to_python_timedelta(
    value: Union[int, float], tu: Optional[str] = "ns"
) -> timedelta:
    if tu == "ns":
        return timedelta(microseconds=value // 1e3)
    elif tu == "us":
        return timedelta(microseconds=value)
    elif tu == "ms":
        return timedelta(milliseconds=value)
    else:
        raise ValueError(f"time unit: {tu} not expected")


def _prepare_row_count_args(
    row_count_name: Optional[str] = None,
    row_count_offset: int = 0,
) -> Optional[Tuple[str, int]]:

    if row_count_name is not None:
        return (row_count_name, row_count_offset)
    else:
        return None


EPOCH = datetime(1970, 1, 1).replace(tzinfo=None)


def _to_python_datetime(
    value: Union[int, float],
    dtype: Type[DataType],
    tu: Optional[str] = "ns",
    tz: Optional["str"] = None,
) -> Union[date, datetime]:
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


def format_path(path: Union[str, Path]) -> str:
    """
    Returns a string path, expanding the home directory if present.
    """
    return os.path.expanduser(path)
