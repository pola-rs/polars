import ctypes
import typing as tp
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

import polars as pl

__all__ = [
    "coerce_arrow",
    "_process_null_values",
    "_ptr_to_numpy",
]


def coerce_arrow(array: "pa.Array") -> "pa.Array":
    # also coerces timezone to naive representation
    # units are accounted for by pyarrow
    if "timestamp" in str(array.type):
        warnings.warn(
            "Conversion of (potentially) timezone aware to naive datetimes. TZ information may be lost",
        )
        ts_ms = pa.compute.cast(array, pa.timestamp("ms"), safe=False)
        ms = pa.compute.cast(ts_ms, pa.int64())
        del ts_ms
        array = pa.compute.cast(ms, pa.timestamp("ms"))
        del ms
    # note: Decimal256 could not be cast to float
    elif isinstance(array.type, pa.Decimal128Type):
        array = pa.compute.cast(array, pa.float64())

    if hasattr(array, "num_chunks") and array.num_chunks > 1:
        # we have to coerce before combining chunks, because pyarrow panics if
        # offsets overflow
        if pa.types.is_string(array.type):
            array = pa.compute.cast(array, pa.large_utf8())
        elif pa.types.is_list(array.type):
            # pyarrow does not seem to support casting from list to largelist
            # so we use convert to large list ourselves and do the re-alloc on polars/arrow side
            chunks = []
            for arr in array.iterchunks():
                chunks.append(pl.from_arrow(arr).to_arrow())
            array = pa.chunked_array(chunks)

        array = array.combine_chunks()
    return array


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
