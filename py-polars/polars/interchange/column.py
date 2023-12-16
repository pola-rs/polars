from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import Categorical
from polars.interchange.buffer import PolarsBuffer
from polars.interchange.protocol import (
    Column,
    ColumnNullType,
    CopyNotAllowedError,
    DtypeKind,
    Endianness,
)
from polars.interchange.utils import polars_dtype_to_dtype
from polars.utils._wrap import wrap_s

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from polars import Series
    from polars.interchange.protocol import CategoricalDescription, ColumnBuffers, Dtype


class PolarsColumn(Column):
    """
    A column object backed by a Polars Series.

    Parameters
    ----------
    column
        The Polars Series backing the column object.
    allow_copy
        Allow data to be copied during operations on this column. If set to `False`,
        a RuntimeError will be raised if data would be copied.

    """

    def __init__(self, column: Series, *, allow_copy: bool = True):
        if column.dtype == Categorical and not column.cat.is_local():
            if not allow_copy:
                raise CopyNotAllowedError(
                    f"column {column.name!r} must be converted to a local categorical"
                )
            column = column.cat.to_local()

        self._col = column
        self._allow_copy = allow_copy

    def size(self) -> int:
        """Size of the column in elements."""
        return self._col.len()

    @property
    def offset(self) -> int:
        """Offset of the first element with respect to the start of the underlying buffer."""  # noqa: W505
        offset, _length, _pointer = self._col._s.get_ptr()
        return offset

    @property
    def dtype(self) -> Dtype:
        """Data type of the column."""
        pl_dtype = self._col.dtype
        return polars_dtype_to_dtype(pl_dtype)

    @property
    def describe_categorical(self) -> CategoricalDescription:
        """
        Description of the categorical data type of the column.

        Raises
        ------
        TypeError
            If the data type of the column is not categorical.

        """
        if self.dtype[0] != DtypeKind.CATEGORICAL:
            raise TypeError("`describe_categorical` only works on categorical columns")

        categories = self._col.cat.get_categories()
        return {
            "is_ordered": not self._col.cat.uses_lexical_ordering(),
            "is_dictionary": True,
            "categories": PolarsColumn(categories, allow_copy=self._allow_copy),
        }

    @property
    def describe_null(self) -> tuple[ColumnNullType, int | None]:
        """Description of the null representation the column uses."""
        if self.null_count == 0:
            return ColumnNullType.NON_NULLABLE, None
        else:
            return ColumnNullType.USE_BITMASK, 0

    @property
    def null_count(self) -> int:
        """The number of null elements."""
        return self._col.null_count()

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata for the column."""
        return {}

    def num_chunks(self) -> int:
        """Return the number of chunks the column consists of."""
        return self._col.n_chunks()

    def get_chunks(self, n_chunks: int | None = None) -> Iterator[PolarsColumn]:
        """
        Return an iterator yielding the column chunks.

        Parameters
        ----------
        n_chunks
            The number of chunks to return. Must be a multiple of the number of chunks
            in the column.

        Notes
        -----
        When `n_chunks` is higher than the number of chunks in the column, a slice
        must be performed that is not on the chunk boundary. This will trigger some
        compute if the column contains null values or if the column is of data type
        boolean.

        """
        total_n_chunks = self.num_chunks()
        chunks = self._col.get_chunks()

        if (n_chunks is None) or (n_chunks == total_n_chunks):
            for chunk in chunks:
                yield PolarsColumn(chunk, allow_copy=self._allow_copy)

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
                    yield PolarsColumn(
                        chunk[start : start + step], allow_copy=self._allow_copy
                    )

    def get_buffers(self) -> ColumnBuffers:
        """Return a dictionary containing the underlying buffers."""
        return {
            "data": self._get_data_buffer(),
            "validity": self._get_validity_buffer(),
            "offsets": self._get_offsets_buffer(),
        }

    def _get_data_buffer(self) -> tuple[PolarsBuffer, Dtype]:
        s = wrap_s(self._col._s.get_buffer(0))
        buffer = PolarsBuffer(s, allow_copy=self._allow_copy)

        dtype = self.dtype
        if dtype[0] == DtypeKind.CATEGORICAL:
            dtype = (DtypeKind.UINT, 32, "I", Endianness.NATIVE)

        return buffer, dtype

    def _get_validity_buffer(self) -> tuple[PolarsBuffer, Dtype] | None:
        buffer = self._col._s.get_buffer(1)
        if buffer is None:
            return None

        s = wrap_s(buffer)
        buffer = PolarsBuffer(s, allow_copy=self._allow_copy)
        dtype = (DtypeKind.BOOL, 1, "b", Endianness.NATIVE)
        return buffer, dtype

    def _get_offsets_buffer(self) -> tuple[PolarsBuffer, Dtype] | None:
        buffer = self._col._s.get_buffer(2)
        if buffer is None:
            return None

        s = wrap_s(buffer)
        buffer = PolarsBuffer(s, allow_copy=self._allow_copy)
        dtype = (DtypeKind.INT, 64, "l", Endianness.NATIVE)
        return buffer, dtype
