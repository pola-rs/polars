from __future__ import annotations

from typing import Any, Iterable, Optional, TypedDict

import polars as pl
from polars.interchange._utils import (
    _IXCategoricalDescription,
    _IXDtypeKind,
    _IXNullKind,
)
from polars.interchange.buffer import _IXBuffer

# TODO: arrow
_IXNULL_DESCRIPTIONS = {
    _IXDtypeKind.FLOAT: (_IXNullKind.USE_BITMASK, 1),
    _IXDtypeKind.DATETIME: (_IXNullKind.USE_BITMASK, 1),
    _IXDtypeKind.INT: (_IXNullKind.USE_BITMASK, 1),
    _IXDtypeKind.UINT: (_IXNullKind.USE_BITMASK, 1),
    _IXDtypeKind.BOOL: (_IXNullKind.USE_BITMASK, 1),
    _IXDtypeKind.CATEGORICAL: (_IXNullKind.USE_BITMASK, 1),
    _IXDtypeKind.STRING: (_IXNullKind.USE_BITMASK, 1),
}


class _IXColumnBuffers(TypedDict):
    """See ``get_buffers`` for more."""

    data: tuple[_IXBuffer, Any]
    validity: Optional[tuple[_IXBuffer, Any]]
    offsets: Optional[tuple[_IXBuffer, Any]]


class _IXColumn:
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).

    TBD: Arrow has a separate "null" dtype, and has no separate mask concept.
         Instead, it seems to use "children" for both columns with a bit mask,
         and for nested dtypes. Unclear whether this is elegant or confusing.
         This design requires checking the null representation explicitly.

         The Arrow design requires checking:
         1. the ARROW_FLAG_NULLABLE (for sentinel values)
         2. if a column has two children, combined with one of those children
            having a null dtype.

         Making the mask concept explicit seems useful. One null dtype would
         not be enough to cover both bit and byte masks, so that would mean
         even more checking if we did it the Arrow way.

    TBD: there's also the "chunk" concept here, which is implicit in Arrow as
         multiple buffers per array (= column here). Semantically it may make
         sense to have both: chunks were meant for example for lazy evaluation
         of data which doesn't fit in memory, while multiple buffers per column
         could also come from doing a selection operation on a single
         contiguous buffer.

         Given these concepts, one would expect chunks to be all of the same
         size (say a 10,000 row dataframe could have 10 chunks of 1,000 rows),
         while multiple buffers could have data-dependent lengths. Not an issue
         in libraries if one column is backed by a single NumPy array, but in
         Arrow it seems possible.
         Are multiple chunks *and* multiple buffers per column necessary for
         the purposes of this interchange protocol, or must producers either
         reuse the chunk concept for this or copy the data?

    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.

    """

    def __init__(self, column: pl.Series, allow_copy: bool = True) -> None:
        if not isinstance(column, pl.Series):
            raise ValueError("`column` is not a Polars Series")

        self._col = column
        self._allow_copy = allow_copy

    @property
    def size(self) -> Optional[int]:
        """
        Size of the column, in elements.

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.
        """
        # TODO: by chunk
        return self._col.len()

    @property
    def offset(self) -> int:
        """
        Offset of first element.

        May be > 0 if using chunks; for example for a column with N chunks of
        equal size M (only the last chunk may be shorter),
        ``offset = n * M``, ``n = 0 .. N-1``.
        """
        # TODO: how is this anything other than zero given
        # offset = n * M if n contains the *first element*?
        ...

    @property
    def dtype(self) -> tuple[_IXDtypeKind, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``.

        Kind :

            - INT = 0`
            - UINT = 1
            - FLOAT = 2
            - BOOL = 20
            - STRING = 21   # UTF-8
            - DATETIME = 22
            - CATEGORICAL = 23

        Bit-width : the number of bits as an integer
        Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        Endianness : current only native endianness (``=``) is supported

        Notes:

            - Kind specifiers are aligned with DLPack where possible (hence the
              jump to 20, leave enough room for future extension)
            - Masks must be specified as boolean with either bit width 1 (for bit
              masks) or 8 (for byte masks).
            - Dtype width in bits was preferred over bytes
            - Endianness isn't too useful, but included now in case in the future
              we need to support non-native endianness
            - Went with Apache Arrow format strings over NumPy format strings
              because they're more complete from a dataframe perspective
            - Format strings are mostly useful for datetime specification, and
              for categoricals.
            - For categoricals, the format string describes the type of the
              categorical in the data buffer. In case of a separate encoding of
              the categorical (e.g. an integer to string mapping), this can
              be derived from ``self.describe_categorical``.
            - Data types not included: complex, Arrow-style null, binary, decimal,
              and nested (list, struct, map, union) dtypes.
        """
        # dtype = self._col.dtype

        # TODO: handle categorical

        # TODO: handle string (don't implement)
        # if is_string_dtype(dtype):
        #       raise NotImplementedError("String data types are not supported")
        ...

    @property
    def describe_categorical(self) -> _IXCategoricalDescription:
        """
        If the dtype is categorical, there are two options:

        - There are only values in the data buffer.
        - There is a separate dictionary-style encoding for categorical values.

        Raises RuntimeError if the dtype is not categorical

        Content of returned dict:

            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a dictionary-style mapping of
                                categorical values to other objects exists
            - "mapping" : dict, Python-level only (e.g. ``{int: str}``).
                          None if not a dictionary-style categorical.

        TBD: are there any other in-memory representations that are needed?
        """
        if self.dtype[0] != _IXDtypeKind.CATEGORICAL:
            raise TypeError("Only categorical types are supported")

        # TODO
        return {"is_ordered": False, "is_dictionary": False, "mapping": None}

    @property
    def describe_null(self) -> tuple[_IXNullKind, Any]:
        """
        Return the missing value (or "null") representation the column dtype
        uses, as a tuple ``(kind, value)``.

        Kind:

            - 0 : non-nullable
            - 1 : NaN/NaT
            - 2 : sentinel value
            - 3 : bit mask
            - 4 : byte mask

        Value : if kind is "sentinel value", the actual value. If kind is a bit
        mask or a byte mask, the value (0 or 1) indicating a missing value. None
        otherwise.
        """
        kind = self.dtype[0]
        if kind not in _IXNULL_DESCRIPTIONS:
            raise NotImplementedError(f"{type(kind)} is not supported")

        null, value = _IXNULL_DESCRIPTIONS[kind]

        return null, value

    # TODO: why use cache here?
    @property
    def null_count(self) -> int:
        """
        Number of null elements, if known.

        Note: Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """
        return self._col.null_count()

    @property
    def metadata(self) -> dict[str, Any]:
        """
        The metadata for the column. See `DataFrame.metadata` for more details.
        """
        return dict()

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
        return self._col.n_chunks()

    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable[_IXColumn]:
        """
        Return an iterator yielding the chunks.

        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        _n_chunks = self._col.n_chunks()

        # return by chunk
        if not n_chunks or n_chunks == _n_chunks:
            chunk_lenghts = self._col.chunk_lengths()
            for i, _len in enumerate(chunk_lenghts):
                yield self._col[i * _len : (i + 1) * _len]

        # TODO: return subdivided chunks
        raise NotImplementedError("Chunk subdividing is not currently supported")

    def get_buffers(
        self,
    ) -> _IXColumnBuffers:
        """
        Return a dictionary containing the underlying buffers.

        The returned dictionary has the following contents:

            - "data": a two-element tuple whose first element is a buffer
                      containing the data and whose second element is the data
                      buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
                          containing mask values indicating missing data and
                          whose second element is the mask value buffer's
                          associated dtype. None if the null representation is
                          not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
                         containing the offset values for variable-size binary
                         data (e.g., variable-length strings) and whose second
                         element is the offsets buffer's associated dtype. None
                         if the data buffer does not have an associated offsets
                         buffer.
        """
        ...
