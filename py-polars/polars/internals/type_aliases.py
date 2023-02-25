from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from polars import internals as pli
    from polars.dependencies import numpy as np
    from polars.dependencies import pandas as pd
    from polars.dependencies import pyarrow as pa


# Types that qualify as expressions (eg: for use in 'select', 'with_columns'...)
PolarsExprType: TypeAlias = "pli.Expr | pli.WhenThen | pli.WhenThenThen"

# literal types that are allowed in expressions (auto-converted to pl.lit)
PythonLiteral: TypeAlias = Union[
    str, int, float, bool, date, time, datetime, timedelta, bytes, Decimal
]

IntoExpr: TypeAlias = "PolarsExprType | PythonLiteral | pli.Series | None"
ComparisonOperator: TypeAlias = Literal["eq", "neq", "gt", "lt", "gt_eq", "lt_eq"]

# User-facing string literal types
# The following all have an equivalent Rust enum with the same name
AvroCompression: TypeAlias = Literal["uncompressed", "snappy", "deflate"]
CategoricalOrdering: TypeAlias = Literal["physical", "lexical"]
CsvEncoding: TypeAlias = Literal["utf8", "utf8-lossy"]
FillNullStrategy: TypeAlias = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
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
StartBy: TypeAlias = Literal["window", "datapoint", "monday"]
TimeUnit: TypeAlias = Literal["ns", "us", "ms"]
UniqueKeepStrategy: TypeAlias = Literal["first", "last", "none"]
UnstackDirection: TypeAlias = Literal["vertical", "horizontal"]

# The following have a Rust enum equivalent with a different name
AsofJoinStrategy: TypeAlias = Literal["backward", "forward"]  # AsofStrategy
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
ConcatMethod = Literal["vertical", "diagonal", "horizontal"]
EpochTimeUnit = Literal["ns", "us", "ms", "s", "d"]
Orientation: TypeAlias = Literal["col", "row"]
SearchSortedSide: TypeAlias = Literal["any", "left", "right"]
TransferEncoding: TypeAlias = Literal["hex", "base64"]

# type signature for allowed frame init
FrameInitTypes: TypeAlias = "Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | pli.Series] | Sequence[Any] | np.ndarray[Any, Any] | pa.Table | pd.DataFrame | pli.Series"
