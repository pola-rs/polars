from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import (
    Boolean,
    Categorical,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)
from polars.interchange.protocol import DtypeKind, Endianness

if TYPE_CHECKING:
    from polars.datatypes import DataTypeClass
    from polars.interchange.protocol import Dtype
    from polars.type_aliases import PolarsDataType

NE = Endianness.NATIVE

dtype_map: dict[DataTypeClass, Dtype] = {
    Int8: (DtypeKind.INT, 8, "c", NE),
    Int16: (DtypeKind.INT, 16, "s", NE),
    Int32: (DtypeKind.INT, 32, "i", NE),
    Int64: (DtypeKind.INT, 64, "l", NE),
    UInt8: (DtypeKind.UINT, 8, "C", NE),
    UInt16: (DtypeKind.UINT, 16, "S", NE),
    UInt32: (DtypeKind.UINT, 32, "I", NE),
    UInt64: (DtypeKind.UINT, 64, "L", NE),
    Float32: (DtypeKind.FLOAT, 32, "f", NE),
    Float64: (DtypeKind.FLOAT, 64, "g", NE),
    Boolean: (DtypeKind.BOOL, 1, "b", NE),
    Utf8: (DtypeKind.STRING, 8, "U", NE),
    Date: (DtypeKind.DATETIME, 32, "tdD", NE),
    Time: (DtypeKind.DATETIME, 64, "ttu", NE),
    Datetime: (DtypeKind.DATETIME, 64, "tsu:", NE),
    Duration: (DtypeKind.DATETIME, 64, "tDu", NE),
    Categorical: (DtypeKind.CATEGORICAL, 32, "I", NE),
}


def polars_dtype_to_dtype(dtype: PolarsDataType) -> Dtype:
    """Convert Polars data type to interchange protocol data type."""
    try:
        result = dtype_map[dtype.base_type()]
    except KeyError as exc:
        raise ValueError(
            f"data type {dtype!r} not supported by the interchange protocol"
        ) from exc

    # Handle instantiated data types
    if isinstance(dtype, Datetime):
        return _datetime_to_dtype(dtype)
    elif isinstance(dtype, Duration):
        return _duration_to_dtype(dtype)

    return result


def _datetime_to_dtype(dtype: Datetime) -> Dtype:
    tu = dtype.time_unit[0] if dtype.time_unit is not None else "u"
    tz = dtype.time_zone if dtype.time_zone is not None else ""
    arrow_c_type = f"ts{tu}:{tz}"
    return DtypeKind.DATETIME, 64, arrow_c_type, NE


def _duration_to_dtype(dtype: Duration) -> Dtype:
    tu = dtype.time_unit[0] if dtype.time_unit is not None else "u"
    arrow_c_type = f"tD{tu}"
    return DtypeKind.DATETIME, 64, arrow_c_type, NE
