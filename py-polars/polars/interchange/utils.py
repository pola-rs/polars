from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.interchange.protocol import DtypeKind, Endianness

if TYPE_CHECKING:
    from polars.interchange.protocol import Dtype
    from polars.type_aliases import PolarsDataType


def polars_dtype_to_dtype(dtype: PolarsDataType) -> Dtype:
    """Convert Polars data type to interchange protocol data type."""
    if dtype == pl.Int8:
        return DtypeKind.INT, 8, "c", Endianness.NATIVE
    elif dtype == pl.Int16:
        return DtypeKind.INT, 16, "s", Endianness.NATIVE
    elif dtype == pl.Int32:
        return DtypeKind.INT, 32, "i", Endianness.NATIVE
    elif dtype == pl.Int64:
        return DtypeKind.INT, 64, "l", Endianness.NATIVE
    elif dtype == pl.UInt8:
        return DtypeKind.UINT, 8, "C", Endianness.NATIVE
    elif dtype == pl.UInt16:
        return DtypeKind.UINT, 16, "S", Endianness.NATIVE
    elif dtype == pl.UInt32:
        return DtypeKind.UINT, 32, "I", Endianness.NATIVE
    elif dtype == pl.UInt64:
        return DtypeKind.UINT, 64, "L", Endianness.NATIVE
    elif dtype == pl.Float32:
        return DtypeKind.FLOAT, 32, "f", Endianness.NATIVE
    elif dtype == pl.Float64:
        return DtypeKind.FLOAT, 64, "g", Endianness.NATIVE
    elif dtype == pl.Boolean:
        return DtypeKind.BOOL, 1, "b", Endianness.NATIVE
    elif dtype == pl.Utf8:
        return DtypeKind.STRING, 8, "U", Endianness.NATIVE
    elif dtype == pl.Date:
        return DtypeKind.DATETIME, 32, "tdD", Endianness.NATIVE
    elif dtype == pl.Time:
        return DtypeKind.DATETIME, 64, "ttu", Endianness.NATIVE
    elif dtype == pl.Datetime:
        tu = dtype.time_unit[0] if dtype.time_unit is not None else "u"  # type: ignore[union-attr]
        tz = dtype.time_zone if dtype.time_zone is not None else ""  # type: ignore[union-attr]
        arrow_c_type = f"ts{tu}:{tz}"
        return DtypeKind.DATETIME, 64, arrow_c_type, Endianness.NATIVE
    elif dtype == pl.Duration:
        tu = dtype.time_unit[0] if dtype.time_unit is not None else "u"  # type: ignore[union-attr]
        arrow_c_type = f"tD{tu}"
        return DtypeKind.DATETIME, 64, arrow_c_type, Endianness.NATIVE
    elif dtype == pl.Categorical:
        return DtypeKind.CATEGORICAL, 32, "I", Endianness.NATIVE
    else:
        raise ValueError(f"data type {dtype} not supported by interchange protocol")
