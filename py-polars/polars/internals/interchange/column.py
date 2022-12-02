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

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    import polars as pl
    from polars.internals.interchange.dataframe_protocol import Dtype


_NO_VALIDITY_BUFFER = {
    ColumnNullType.NON_NULLABLE: "This column is non-nullable",
    ColumnNullType.USE_NAN: "This column uses NaN as null",
    ColumnNullType.USE_SENTINEL: "This column uses a sentinel value",
}


class PolarsColumn(Column):
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.
    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).
    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.
    """

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
        dtype = self._col.dtype
        if dtype == pl.Int8:
            return DtypeKind.INT, 8, "c", "="
        elif dtype == pl.Int16:
            return DtypeKind.INT, 16, "s", "="
        elif dtype == pl.Int32:
            return DtypeKind.INT, 32, "i", "="
        elif dtype == pl.Int64:
            return DtypeKind.INT, 64, "l", "="
        elif dtype == pl.UInt8:
            return DtypeKind.UINT, 8, "C", "="
        elif dtype == pl.UInt16:
            return DtypeKind.UINT, 16, "S", "="
        elif dtype == pl.UInt32:
            return DtypeKind.UINT, 32, "I", "="
        elif dtype == pl.UInt64:
            return DtypeKind.UINT, 64, "L", "="
        elif dtype == pl.Float32:
            return DtypeKind.FLOAT, 32, "f", "="
        elif dtype == pl.Float64:
            return DtypeKind.FLOAT, 64, "g", "="
        elif dtype == pl.Boolean:
            return DtypeKind.BOOL, 8, "b", "="
        elif dtype == pl.Utf8:
            return DtypeKind.STRING, 64, "U", "="
        elif dtype == pl.Date:
            return DtypeKind.DATETIME, 32, "tdD", "="
        elif dtype == pl.Time:
            return DtypeKind.DATETIME, 64, "ttu", "="
        elif dtype == pl.Datetime:
            tu = dtype.tu[0] if dtype.tu is not None else "u"
            tz = dtype.tz if dtype.tz is not None else ""
            arrow_c_type = f"ts{tu}:{tz}"
            return DtypeKind.DATETIME, 64, arrow_c_type, "="
        elif dtype == pl.Duration:
            tu = dtype.tu[0] if dtype.tu is not None else "u"
            arrow_c_type = f"tD{tu}"
            return DtypeKind.DATETIME, 64, arrow_c_type, "="
        elif dtype == pl.Categorical:
            return DtypeKind.CATEGORICAL, 32, "I", "="
        else:
            raise ValueError(f"Data type {dtype} not supported by interchange protocol")

    @property
    def describe_categorical(self) -> CategoricalDescription:
        if self.dtype[0] != DtypeKind.CATEGORICAL:
            raise TypeError(
                "describe_categorical only works on a column with categorical dtype!"
            )

        return {
            "is_ordered": self._col.cat.ordered,  # TODO: Implement
            "is_dictionary": True,
            "categories": PolarsColumn(self._col.cat.categories),  # TODO: Implement
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
        # TODO
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
        if self.dtype[0] == DtypeKind.CATEGORICAL:
            codes = self._col.to_physical()
            buffer = PolarsBuffer(codes, allow_copy=self._allow_copy)
            dtype = (DtypeKind.UINT, 32, "I", "=")
        elif self.dtype[0] == DtypeKind.STRING:
            # TODO
            # Marshal the strings from a NumPy object array into a byte array
            buf = self._col.to_numpy()
            b = bytearray()

            # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
            for obj in buf:
                if isinstance(obj, str):
                    b.extend(obj.encode(encoding="utf-8"))

            # Convert the byte array to a Pandas "buffer" using
            # a NumPy array as the backing store
            buffer = PolarsBuffer(np.frombuffer(b, dtype="uint8"))

            # Define the dtype for the returned buffer
            dtype = (
                DtypeKind.STRING,
                8,
                ArrowCTypes.STRING,
                Endianness.NATIVE,
            )
        else:
            buffer = PolarsBuffer(self._col, allow_copy=self._allow_copy)
            dtype = self.dtype

        return buffer, dtype

    def _get_validity_buffer(self) -> tuple[PolarsBuffer, Dtype]:
        """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.
        """
        buffer = PolarsBuffer(self._col.is_not_null(), self._allow_copy)
        dtype = (DtypeKind.BOOL, 8, "b", "=")
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

        # For each string, we need to manually determine the next offset
        values = self._col.to_numpy()
        ptr = 0
        offsets = np.zeros(shape=(len(values) + 1,), dtype=np.int64)
        for i, v in enumerate(values):
            # For missing values (in this case, `np.nan` values)
            # we don't increment the pointer
            if isinstance(v, str):
                b = v.encode(encoding="utf-8")
                ptr += len(b)

            offsets[i + 1] = ptr

        # Convert the offsets to a Pandas "buffer" using
        # the NumPy array as the backing store
        buffer = PolarsBuffer(offsets)

        # Assemble the buffer dtype info
        dtype = (
            DtypeKind.INT,
            64,
            ArrowCTypes.INT64,
            Endianness.NATIVE,
        )  # note: currently only support native endianness

        return buffer, dtype
