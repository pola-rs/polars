from __future__ import annotations

from typing import TYPE_CHECKING

from polars.internals.interchange.dataframe_protocol import DtypeKind

if TYPE_CHECKING:
    import polars as pl
    from polars.internals.interchange.dataframe_protocol import Dtype

NATIVE_ENDIANNESS = "="


def polars_dtype_to_dtype(dtype: pl.DataType) -> Dtype:
    if dtype == pl.Int8:
        return DtypeKind.INT, 8, "c", NATIVE_ENDIANNESS
    elif dtype == pl.Int16:
        return DtypeKind.INT, 16, "s", NATIVE_ENDIANNESS
    elif dtype == pl.Int32:
        return DtypeKind.INT, 32, "i", NATIVE_ENDIANNESS
    elif dtype == pl.Int64:
        return DtypeKind.INT, 64, "l", NATIVE_ENDIANNESS
    elif dtype == pl.UInt8:
        return DtypeKind.UINT, 8, "C", NATIVE_ENDIANNESS
    elif dtype == pl.UInt16:
        return DtypeKind.UINT, 16, "S", NATIVE_ENDIANNESS
    elif dtype == pl.UInt32:
        return DtypeKind.UINT, 32, "I", NATIVE_ENDIANNESS
    elif dtype == pl.UInt64:
        return DtypeKind.UINT, 64, "L", NATIVE_ENDIANNESS
    elif dtype == pl.Float32:
        return DtypeKind.FLOAT, 32, "f", NATIVE_ENDIANNESS
    elif dtype == pl.Float64:
        return DtypeKind.FLOAT, 64, "g", NATIVE_ENDIANNESS
    elif dtype == pl.Boolean:
        return DtypeKind.BOOL, 8, "b", NATIVE_ENDIANNESS
    elif dtype == pl.Utf8:
        return DtypeKind.STRING, 64, "U", NATIVE_ENDIANNESS
    elif dtype == pl.Date:
        return DtypeKind.DATETIME, 32, "tdD", NATIVE_ENDIANNESS
    elif dtype == pl.Time:
        return DtypeKind.DATETIME, 64, "ttu", NATIVE_ENDIANNESS
    elif dtype == pl.Datetime:
        tu = dtype.tu[0] if dtype.tu is not None else "u"
        tz = dtype.tz if dtype.tz is not None else ""
        arrow_c_type = f"ts{tu}:{tz}"
        return DtypeKind.DATETIME, 64, arrow_c_type, NATIVE_ENDIANNESS
    elif dtype == pl.Duration:
        tu = dtype.tu[0] if dtype.tu is not None else "u"
        arrow_c_type = f"tD{tu}"
        return DtypeKind.DATETIME, 64, arrow_c_type, NATIVE_ENDIANNESS
    elif dtype == pl.Categorical:
        return DtypeKind.CATEGORICAL, 32, "I", NATIVE_ENDIANNESS
    else:
        raise ValueError(f"Data type {dtype} not supported by interchange protocol")
