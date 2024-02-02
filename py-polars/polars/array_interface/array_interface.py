from __future__ import annotations

from typing import TYPE_CHECKING

from polars.array_interface.protocol import PROTOCOL_VERSION
from polars.array_interface.utils import dtype_to_typestr

if TYPE_CHECKING:
    from polars import Series
    from polars.array_interface.protocol import ArrayInterface


def series_array_interface(s: Series) -> ArrayInterface:
    """..."""
    typestr = dtype_to_typestr(s.dtype)
    buffers = s._get_buffers()

    pointer, offset, length = buffers["values"]._get_buffer_info()

    mask = buffers["validity"] if s.null_count() > 0 else None

    return {
        "shape": (length,),
        "typestr": typestr,
        "descr": [("", typestr)],
        "data": (pointer, True),
        "strides": None,
        "mask": mask,
        "offset": offset,
        "version": PROTOCOL_VERSION,
    }
