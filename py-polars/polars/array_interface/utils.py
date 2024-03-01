from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import (
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType

_dtype_to_typestr_map: dict[PolarsDataType, str] = {
    Int8: "|i1",
    Int16: "<i2",
    Int32: "<i4",
    Int64: "<i8",
    UInt8: "|u1",
    UInt16: "<u2",
    UInt32: "<u4",
    UInt64: "<u8",
    Float32: "<f4",
    Float64: "<f8",
}


def dtype_to_typestr(dtype: PolarsDataType) -> str:
    """Convert a Polars datatype to a type string."""
    # TODO: Handle non-primitives
    return _dtype_to_typestr_map[dtype]
