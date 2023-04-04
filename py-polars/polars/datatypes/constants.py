from __future__ import annotations

from typing import TYPE_CHECKING

from polars.datatypes import (
    Binary,
    Boolean,
    Categorical,
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
    List,
    Null,
    Object,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)

if TYPE_CHECKING:
    from polars.type_aliases import PolarsDataType, TimeUnit

DTYPE_TEMPORAL_UNITS: frozenset[TimeUnit] = frozenset(["ns", "us", "ms"])

DATETIME_DTYPES: frozenset[PolarsDataType] = frozenset(
    [
        # TODO: ideally need a mechanism to wildcard timezones here too
        Datetime("ms"),
        Datetime("us"),
        Datetime("ns"),
    ]
)
DURATION_DTYPES: frozenset[PolarsDataType] = frozenset(
    [
        Duration("ms"),
        Duration("us"),
        Duration("ns"),
    ]
)
TEMPORAL_DTYPES: frozenset[PolarsDataType] = (
    frozenset([Date, Time]) | DATETIME_DTYPES | DURATION_DTYPES
)

INTEGER_DTYPES: frozenset[PolarsDataType] = frozenset(
    [
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
    ]
)
FLOAT_DTYPES: frozenset[PolarsDataType] = frozenset([Float32, Float64])
NUMERIC_DTYPES: frozenset[PolarsDataType] = (
    FLOAT_DTYPES | INTEGER_DTYPES | frozenset([Decimal])
)

NESTED_DTYPES: frozenset[PolarsDataType] = frozenset([List, Struct])

POLARS_DTYPES: frozenset[PolarsDataType] = (
    NUMERIC_DTYPES
    # Not using instantiated Datetime/Duration types here
    | frozenset([Date, Time, Duration, Datetime])
    | NESTED_DTYPES
    | frozenset([Boolean, Utf8, Binary, Categorical, Null, Object])
)

# number of rows to scan by default when inferring datatypes
N_INFER_DEFAULT = 100
