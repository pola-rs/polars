import ctypes
import typing as tp
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Tuple, Union

import numpy as np


def _process_null_values(
    null_values: Union[None, str, tp.List[str], Dict[str, str]] = None,
) -> Union[None, str, tp.List[str], tp.List[Tuple[str, str]]]:
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
