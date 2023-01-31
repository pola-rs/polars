from __future__ import annotations

import sys

from polars import internals as pli

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

IntoExpr: TypeAlias = "int | float | str | pli.Expr | pli.Series"
ComparisonOperator: TypeAlias = Literal["eq", "neq", "gt", "lt", "gt_eq", "lt_eq"]

# User-facing string literal types
# The following all have an equivalent Rust enum with the same name
AvroCompression: TypeAlias = Literal["uncompressed", "snappy", "deflate"]
CategoricalOrdering: TypeAlias = Literal["physical", "lexical"]
StartBy: TypeAlias = Literal["window", "datapoint", "monday"]
CsvEncoding: TypeAlias = Literal["utf8", "utf8-lossy"]
FillNullStrategy: TypeAlias = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
IpcCompression: TypeAlias = Literal["uncompressed", "lz4", "zstd"]
NullBehavior: TypeAlias = Literal["ignore", "drop"]
NullStrategy: TypeAlias = Literal["ignore", "propagate"]
UnstackDirection: TypeAlias = Literal["vertical", "horizontal"]
ParallelStrategy: TypeAlias = Literal["auto", "columns", "row_groups", "none"]
ParquetCompression: TypeAlias = Literal[
    "lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"
]
PivotAgg: TypeAlias = Literal[
    "first", "sum", "max", "min", "mean", "median", "last", "count"
]
RankMethod: TypeAlias = Literal["average", "min", "max", "dense", "ordinal", "random"]
TimeUnit: TypeAlias = Literal["ns", "us", "ms"]
UniqueKeepStrategy: TypeAlias = Literal["first", "last", "none"]
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

# The following have a Rust enum equivalent with a different name
AsofJoinStrategy: TypeAlias = Literal["backward", "forward"]  # AsofStrategy
ClosedInterval: TypeAlias = Literal["left", "right", "both", "none"]  # ClosedWindow
RollingInterpolationMethod: TypeAlias = Literal[
    "nearest", "higher", "lower", "midpoint", "linear"
]  # QuantileInterpolOptions
JoinStrategy: TypeAlias = Literal[
    "inner", "left", "outer", "semi", "anti", "cross"
]  # JoinType
ToStructStrategy: TypeAlias = Literal[
    "first_non_null", "max_width"
]  # ListToStructWidthStrategy
InterpolationMethod: TypeAlias = Literal["linear", "nearest"]

# The following have no equivalent on the Rust side
ConcatMethod = Literal["vertical", "diagonal", "horizontal"]
EpochTimeUnit = Literal["ns", "us", "ms", "s", "d"]
Orientation: TypeAlias = Literal["col", "row"]
TransferEncoding: TypeAlias = Literal["hex", "base64"]
SearchSortedSide: TypeAlias = Literal["any", "left", "right"]
