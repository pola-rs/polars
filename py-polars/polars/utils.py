import ctypes
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

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


def _datetime_to_pl_timestamp(dt: datetime) -> int:
    """
    Converts a python datetime to a timestamp in nanoseconds
    """
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1e9)


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
