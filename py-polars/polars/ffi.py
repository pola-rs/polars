import numpy as np
from numpy import ctypeslib
import ctypes
from typing import Any


def ptr_to_numpy(ptr: int, len: int, ptr_type: Any) -> np.ndarray:
    """

    Parameters
    ----------
    ptr
        C/Rust ptr casted to usize
    len
        Lenght of the array values
    ptr_type
        Example:
            f32: ctypes.c_float)

    Returns
    -------
    View of memory block as numpy array

    """
    ptr = ctypes.cast(ptr, ctypes.POINTER(ptr_type))
    return ctypeslib.as_array(ptr, (len,))
