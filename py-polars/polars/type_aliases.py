from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from polars import DataFrame, Expr, LazyFrame, Series
    from polars.datatypes import DataType, DataTypeClass, TemporalType
    from polars.dependencies import numpy as np
    from polars.dependencies import pandas as pd
    from polars.dependencies import pyarrow as pa
    from polars.functions.whenthen import WhenThen, WhenThenThen

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

# Data types
PolarsDataType: TypeAlias = Union["DataTypeClass", "DataType"]
PolarsTemporalType: TypeAlias = Union[Type["TemporalType"], "TemporalType"]
OneOrMoreDataTypes: TypeAlias = Union[PolarsDataType, Iterable[PolarsDataType]]
PythonDataType: TypeAlias = Union[
    Type[int],
    Type[float],
    Type[bool],
    Type[str],
    Type[date],
    Type[time],
    Type[datetime],
    Type[timedelta],
    Type[List[Any]],
    Type[Tuple[Any, ...]],
    Type[bytes],
    Type[Decimal],
    Type[None],
]

SchemaDefinition: TypeAlias = Union[
    Sequence[str],
    Mapping[str, Union[PolarsDataType, PythonDataType]],
    Sequence[Union[str, Tuple[str, Union[PolarsDataType, PythonDataType, None]]]],
]
SchemaDict: TypeAlias = Mapping[str, PolarsDataType]

# Types that qualify as expressions (eg: for use in 'select', 'with_columns'...)
PolarsExprType: TypeAlias = Union["Expr", "WhenThen", "WhenThenThen"]

# literal types that are allowed in expressions (auto-converted to pl.lit)
PythonLiteral: TypeAlias = Union[
    str, int, float, bool, date, time, datetime, timedelta, bytes, Decimal, List[Any]
]

IntoExpr: TypeAlias = Union[PolarsExprType, PythonLiteral, "Series", None]
ComparisonOperator: TypeAlias = Literal["eq", "neq", "gt", "lt", "gt_eq", "lt_eq"]

# User-facing string literal types
# The following all have an equivalent Rust enum with the same name
AvroCompression: TypeAlias = Literal["uncompressed", "snappy", "deflate"]
CategoricalOrdering: TypeAlias = Literal["physical", "lexical"]
CsvEncoding: TypeAlias = Literal["utf8", "utf8-lossy"]
FillNullStrategy: TypeAlias = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
FloatFmt: TypeAlias = Literal["full", "mixed"]
IpcCompression: TypeAlias = Literal["uncompressed", "lz4", "zstd"]
NullBehavior: TypeAlias = Literal["ignore", "drop"]
NullStrategy: TypeAlias = Literal["ignore", "propagate"]
ParallelStrategy: TypeAlias = Literal["auto", "columns", "row_groups", "none"]
ParquetCompression: TypeAlias = Literal[
    "lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"
]
PivotAgg: TypeAlias = Literal[
    "first", "sum", "max", "min", "mean", "median", "last", "count"
]
RankMethod: TypeAlias = Literal["average", "min", "max", "dense", "ordinal", "random"]
SizeUnit: TypeAlias = Literal[
    "b",
    "kb",
    "mb",
    "gb",
    "tb",
    "bytes",
    "kilobytes",
    "megabytes",
    "gigabytes",
    "terabytes",
]
StartBy: TypeAlias = Literal[
    "window",
    "datapoint",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
TimeUnit: TypeAlias = Literal["ns", "us", "ms"]
UniqueKeepStrategy: TypeAlias = Literal["first", "last", "any", "none"]
UnstackDirection: TypeAlias = Literal["vertical", "horizontal"]
ApplyStrategy: TypeAlias = Literal["thread_local", "threading"]

# The following have a Rust enum equivalent with a different name
AsofJoinStrategy: TypeAlias = Literal["backward", "forward", "nearest"]  # AsofStrategy
ClosedInterval: TypeAlias = Literal["left", "right", "both", "none"]  # ClosedWindow
InterpolationMethod: TypeAlias = Literal["linear", "nearest"]
JoinStrategy: TypeAlias = Literal[
    "inner", "left", "outer", "semi", "anti", "cross"
]  # JoinType
RollingInterpolationMethod: TypeAlias = Literal[
    "nearest", "higher", "lower", "midpoint", "linear"
]  # QuantileInterpolOptions
ToStructStrategy: TypeAlias = Literal[
    "first_non_null", "max_width"
]  # ListToStructWidthStrategy

# The following have no equivalent on the Rust side
ConcatMethod = Literal["vertical", "diagonal", "horizontal", "align"]
EpochTimeUnit = Literal["ns", "us", "ms", "s", "d"]
Orientation: TypeAlias = Literal["col", "row"]
SearchSortedSide: TypeAlias = Literal["any", "left", "right"]
TransferEncoding: TypeAlias = Literal["hex", "base64"]
CorrelationMethod: TypeAlias = Literal["pearson", "spearman"]
DbReadEngine: TypeAlias = Literal["adbc", "connectorx"]
DbWriteEngine: TypeAlias = Literal["sqlalchemy", "adbc"]
DbWriteMode: TypeAlias = Literal["replace", "append", "fail"]
WindowMappingStrategy: TypeAlias = Literal["group_to_rows", "join", "explode"]

# type signature for allowed frame init
FrameInitTypes: TypeAlias = Union[
    Mapping[str, Union[Sequence[object], Mapping[str, Sequence[object]], "Series"]],
    Sequence[Any],
    "np.ndarray[Any, Any]",
    "pa.Table",
    "pd.DataFrame",
]

# Excel IO
ConditionalFormatDict: TypeAlias = Mapping[
    # dict of colname(s) to str, dict, or sequence of str/dict
    Union[str, Collection[str]],
    Union[str, Union[Mapping[str, Any], Sequence[Union[str, Mapping[str, Any]]]]],
]
ColumnTotalsDefinition: TypeAlias = Union[
    # dict of colname(s) to str, a collection of str, or a boolean
    Mapping[Union[str, Collection[str]], str],
    Sequence[str],
    bool,
]
RowTotalsDefinition: TypeAlias = Union[
    # dict of colname to str(s), a collection of str, or a boolean
    Mapping[str, Union[str, Collection[str]]],
    Collection[str],
    bool,
]

# standard/named hypothesis profiles used for parametric testing
ParametricProfileNames: TypeAlias = Literal["fast", "balanced", "expensive"]

# typevars for core polars types
PolarsType = TypeVar("PolarsType", "DataFrame", "LazyFrame", "Series", "Expr")
FrameType = TypeVar("FrameType", "DataFrame", "LazyFrame")
