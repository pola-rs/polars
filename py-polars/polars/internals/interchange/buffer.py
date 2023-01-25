from __future__ import annotations

import polars as pl
from polars.internals.interchange.dataframe_protocol import (
    Buffer,
    DlpackDeviceType,
    DtypeKind,
)
from polars.internals.interchange.utils import polars_dtype_to_dtype


class PolarsBuffer(Buffer):
    """A buffer represented by a Polars Series consisting of a single chunk."""

    def __init__(self, data: pl.Series, allow_copy: bool = True) -> None:
        if data.n_chunks() > 1:
            if allow_copy:
                data = data.rechunk()
            else:
                raise RuntimeError(
                    "Exports cannot be zero-copy in the case of a non-contiguous buffer"
                )

        self._data = data

    @property
    def bufsize(self) -> int:
        """Buffer size in bytes."""
        dtype = polars_dtype_to_dtype(self._data.dtype)
        bytes_per_element = dtype[1] // 8

        if dtype[0] == DtypeKind.STRING:
            return self._data.str.lengths().sum() * bytes_per_element
        else:
            return len(self._data) * bytes_per_element

    @property
    def ptr(self) -> int:
        """Pointer to start of the buffer as an integer."""
        return self._data._s.get_ptr()

    def __dlpack__(self):
        """Represent this structure as DLPack interface."""
        raise NotImplementedError("__dlpack__")

    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """Device type and device ID for where the data in the buffer resides."""
        return (DlpackDeviceType.CPU, None)

    def __repr__(self) -> str:
        buffer_info = {
            "bufsize": self.bufsize,
            "ptr": self.ptr,
            "device": self.__dlpack_device__()[0].name,
        }
        return f"PolarsBuffer({buffer_info})"
