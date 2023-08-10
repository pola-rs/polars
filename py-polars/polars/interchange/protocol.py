from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Literal, Tuple, TypedDict

if TYPE_CHECKING:
    import sys

    from polars.interchange.buffer import PolarsBuffer
    from polars.interchange.column import PolarsColumn

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias


class DtypeKind(IntEnum):
    """
    Integer enum for data types.

    Attributes
    ----------
    INT : int
        Matches to signed integer data type.
    UINT : int
        Matches to unsigned integer data type.
    FLOAT : int
        Matches to floating point data type.
    BOOL : int
        Matches to boolean data type.
    STRING : int
        Matches to string data type (UTF-8 encoded).
    DATETIME : int
        Matches to datetime data type.
    CATEGORICAL : int
        Matches to categorical data type.
    """

    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


Dtype: TypeAlias = Tuple[DtypeKind, int, str, str]  # see Column.dtype


class ColumnNullType(IntEnum):
    """
    Integer enum for null type representation.

    Attributes
    ----------
    NON_NULLABLE : int
        Non-nullable column.
    USE_NAN : int
        Use explicit float NaN value.
    USE_SENTINEL : int
        Sentinel value besides NaN.
    USE_BITMASK : int
        The bit is set/unset representing a null on a certain position.
    USE_BYTEMASK : int
        The byte is set/unset representing a null on a certain position.
    """

    NON_NULLABLE = 0
    USE_NAN = 1
    USE_SENTINEL = 2
    USE_BITMASK = 3
    USE_BYTEMASK = 4


class ColumnBuffers(TypedDict):
    """Buffers backing a column."""

    # first element is a buffer containing the column data;
    # second element is the data buffer's associated dtype
    data: tuple[PolarsBuffer, Dtype]

    # first element is a buffer containing mask values indicating missing data;
    # second element is the mask value buffer's associated dtype.
    # None if the null representation is not a bit or byte mask
    validity: tuple[PolarsBuffer, Dtype] | None

    # first element is a buffer containing the offset values for
    # variable-size binary data (e.g., variable-length strings);
    # second element is the offsets buffer's associated dtype.
    # None if the data buffer does not have an associated offsets buffer
    offsets: tuple[PolarsBuffer, Dtype] | None


class CategoricalDescription(TypedDict):
    """Description of a categorical column."""

    # whether the ordering of dictionary indices is semantically meaningful
    is_ordered: bool
    # whether a dictionary-style mapping of categorical values to other objects exists
    is_dictionary: Literal[True]
    # Python-level only (e.g. ``{int: str}``).
    # None if not a dictionary-style categorical.
    categories: PolarsColumn


class DlpackDeviceType(IntEnum):
    """Integer enum for device type codes matching DLPack."""

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


class Endianness:
    """Enum indicating the byte-order of a data type."""

    LITTLE = "<"
    BIG = ">"
    NATIVE = "="
    NA = "|"
