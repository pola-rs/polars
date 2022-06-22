from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional, TypedDict

import polars as pl


class _IXDtypeKind(IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


class _IXNullKind(IntEnum):
    NON_NULLABLE = 0
    USE_NAN = 1
    USE_SENTINEL = 2
    USE_BITMASK = 3
    USE_BYTEMASK = 4


class _IXCategoricalDescription(TypedDict):
    """See ``IXColumn.describe_categorical`` for more."""

    is_ordered: bool
    is_dictionary: bool
    mapping: Optional[dict]


class _IXArrowCTypes:
    """
    Enum for Apache Arrow C type format strings.

    The Arrow C data interface:
    https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
    """

    NULL = "n"
    BOOL = "b"
    INT8 = "c"
    UINT8 = "C"
    INT16 = "s"
    UINT16 = "S"
    INT32 = "i"
    UINT32 = "I"
    INT64 = "l"
    UINT64 = "L"
    FLOAT16 = "e"
    FLOAT32 = "f"
    FLOAT64 = "g"
    STRING = "u"  # utf-8
    DATE32 = "tdD"
    DATE64 = "tdm"
    # Resoulution:
    #   - seconds -> 's'
    #   - milliseconds -> 'm'
    #   - microseconds -> 'u'
    #   - nanoseconds -> 'n'
    TIMESTAMP = "ts{resolution}:{tz}"
    TIME = "tt{resolution}"


class _IXEndianness:
    """Enum indicating the byte-order of a data-type."""

    LITTLE = "<"
    BIG = ">"
    NATIVE = "="
    NA = "|"


# TODO: remove Any
def is_string_dtype(dtype: Any) -> bool:
    ...


# TODO: remove Any
def _ix_chunk_to_polars_df(chunk: Any) -> pl.DataFrame:
    pass
