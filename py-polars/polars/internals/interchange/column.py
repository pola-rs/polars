from __future__ import annotations

from typing import TYPE_CHECKING

from polars.internals.interchange.buffer import PolarsBuffer
from polars.internals.interchange.dataframe_protocol import (
    CategoricalDescription,
    Column,
    ColumnBuffers,
    ColumnNullType,
    DtypeKind,
)
from polars.internals.interchange.utils import polars_dtype_to_dtype

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    import polars as pl
    from polars.internals.interchange.dataframe_protocol import Dtype


class PolarsColumn(Column):
    """A column represented by a Polars Series."""

    def __init__(self, column: pl.Series, allow_copy: bool = True):
        self._col = column
        self._allow_copy = allow_copy

    def size(self) -> int:
        """Size of the column, in elements."""
        return len(self._col)

    @property
    def offset(self) -> int:
        return 0

    @property
    def dtype(self) -> Dtype:
        pl_dtype = self._col.dtype
        return polars_dtype_to_dtype(pl_dtype)

    @property
    def describe_categorical(self) -> CategoricalDescription:
        if self.dtype[0] != DtypeKind.CATEGORICAL:
            raise TypeError(
                "describe_categorical only works on a column with categorical dtype!"
            )

        categories = self._col.unique().cat.set_ordering("physical").sort()
        return {
            "is_ordered": self._col.cat.ordered,  # TODO: Implement
            "is_dictionary": True,
            "categories": PolarsColumn(categories),
        }

    @property
    def describe_null(self) -> tuple[ColumnNullType, int]:
        return ColumnNullType.USE_BITMASK, 0

    def null_count(self) -> int:
        return self._col.null_count()

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    def num_chunks(self) -> int:
        return self._col.n_chunks()

    def get_chunks(self, n_chunks: int | None = None) -> Iterable[PolarsColumn]:
        total_n_chunks = self.num_chunks()
        chunks = self._col.get_chunks()  # TODO: Implement

        if (n_chunks is None) or (n_chunks == total_n_chunks):
            for chunk in chunks:
                yield PolarsColumn(chunk, self._allow_copy)

        elif (n_chunks <= 0) or (n_chunks % total_n_chunks != 0):
            raise ValueError(
                "`n_chunks` must be a multiple of the number of chunks of this column"
                f" ({total_n_chunks})"
            )

        else:
            subchunks_per_chunk = n_chunks // total_n_chunks
            for chunk in chunks:
                size = len(chunk)
                step = size // subchunks_per_chunk
                if size % subchunks_per_chunk != 0:
                    step += 1
                for start in range(0, step * subchunks_per_chunk, step):
                    yield PolarsColumn(chunk[start : start + step], self._allow_copy)

    def get_buffers(self) -> ColumnBuffers:
        buffers: ColumnBuffers = {
            "data": self._get_data_buffer(),
            "validity": self._get_validity_buffer(),
            "offsets": None,
        }

        try:
            buffers["offsets"] = self._get_offsets_buffer()
        except TypeError:
            pass

        return buffers

    def _get_data_buffer(self) -> tuple[PolarsBuffer, Dtype]:
        """
        Return the buffer containing the data and the buffer's associated dtype.
        """
        if self.dtype[0] in {
            DtypeKind.INT,
            DtypeKind.UINT,
            DtypeKind.FLOAT,
            DtypeKind.BOOL,
            DtypeKind.DATETIME,
            DtypeKind.STRING,
        }:
            buffer = PolarsBuffer(self._col, allow_copy=self._allow_copy)
            dtype = self.dtype
        elif self.dtype[0] == DtypeKind.CATEGORICAL:
            codes = self._col.to_physical()
            buffer = PolarsBuffer(codes, allow_copy=self._allow_copy)
            dtype = polars_dtype_to_dtype(pl.UInt32)
        else:
            raise NotImplementedError(f"Data type {self._col.dtype} not handled yet")

        return buffer, dtype

    def _get_validity_buffer(self) -> tuple[PolarsBuffer, Dtype]:
        """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.
        """
        buffer = PolarsBuffer(self._col.is_not_null(), self._allow_copy)
        dtype = polars_dtype_to_dtype(pl.Boolean)
        return buffer, dtype

    def _get_offsets_buffer(self) -> tuple[PolarsBuffer, Dtype]:
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.
        Raises NoBufferPresent if the data buffer does not have an associated
        offsets buffer.
        """
        if self.dtype[0] != DtypeKind.STRING:
            raise TypeError(
                "This column has a fixed-length dtype so "
                "it does not have an offsets buffer"
            )

        offsets = (
            self._col.str.n_chars()
            .fill_null(0)
            .cumsum()
            .extend_constant(None, 1)
            .shift_and_fill(1, 0)
            .rechunk()
        )

        buffer = PolarsBuffer(offsets, allow_copy=self._allow_copy)
        dtype = polars_dtype_to_dtype(pl.UInt32)

        return buffer, dtype
