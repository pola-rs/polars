import numpy as np
import numpy.core
from numpy import ctypeslib
import ctypes
from typing import Any

# https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy


def _ptr_to_numpy(ptr: int, len: int, ptr_type: Any) -> np.ndarray:
    """

    Parameters
    ----------
    ptr
        C/Rust ptr casted to usize
    len
        Length of the array values
    ptr_type
        Example:
            f32: ctypes.c_float)

    Returns
    -------
    View of memory block as numpy array

    """
    ptr = ctypes.cast(ptr, ctypes.POINTER(ptr_type))
    return ctypeslib.as_array(ptr, (len,))


def _as_float_ndarray(ptr, size):
    """
    https://github.com/maciejkula/python-rustlearn

    Turn a float* to a numpy array.
    """

    return np.core.multiarray.int_asbuffer(ptr, size * np.float32.itemsize)
