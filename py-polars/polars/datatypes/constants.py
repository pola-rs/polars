from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import (
    DataTypeGroup,
    Date,
    Datetime,
    Decimal,
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
)

if TYPE_CHECKING:
    from polars.datatypes import PolarsDataType
    from polars.type_aliases import TimeUnit


DTYPE_TEMPORAL_UNITS: frozenset[TimeUnit] = frozenset(["ns", "us", "ms"])
DATETIME_DTYPES: frozenset[PolarsDataType] = DataTypeGroup(
    [
        Datetime,
        Datetime("ms"),
        Datetime("us"),
        Datetime("ns"),
    ]
)
DURATION_DTYPES: frozenset[PolarsDataType] = DataTypeGroup(
    [
        Duration,
        Duration("ms"),
        Duration("us"),
        Duration("ns"),
    ]
)
TEMPORAL_DTYPES: frozenset[PolarsDataType] = DataTypeGroup(
    frozenset([Date, Time]) | DATETIME_DTYPES | DURATION_DTYPES
)
SIGNED_INTEGER_DTYPES: frozenset[PolarsDataType] = DataTypeGroup(
    [
        Int8,
        Int16,
        Int32,
        Int64,
    ]
)
UNSIGNED_INTEGER_DTYPES: frozenset[PolarsDataType] = DataTypeGroup(
    [
        UInt8,
        UInt16,
        UInt32,
        UInt64,
    ]
)
INTEGER_DTYPES: frozenset[PolarsDataType] = (
    SIGNED_INTEGER_DTYPES | UNSIGNED_INTEGER_DTYPES
)
FLOAT_DTYPES: frozenset[PolarsDataType] = DataTypeGroup([Float32, Float64])
NUMERIC_DTYPES: frozenset[PolarsDataType] = DataTypeGroup(
    FLOAT_DTYPES | INTEGER_DTYPES | frozenset([Decimal])
)

# number of rows to scan by default when inferring datatypes
N_INFER_DEFAULT = 100
