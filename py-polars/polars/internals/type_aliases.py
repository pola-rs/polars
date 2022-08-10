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
ClosedWindow: TypeAlias = Literal["left", "right", "both", "none"]
FillNullStrategy: TypeAlias = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
NullStrategy: TypeAlias = Literal["ignore", "propagate"]
NullBehavior: TypeAlias = Literal["ignore", "drop"]
ParallelStrategy: TypeAlias = Literal["auto", "columns", "row_groups", "none"]
CsvEncoding: TypeAlias = Literal["utf8", "utf8-lossy"]
AvroCompression: TypeAlias = Literal["uncompressed", "snappy", "deflate"]
IpcCompression: TypeAlias = Literal["uncompressed", "lz4", "zstd"]
ParquetCompression: TypeAlias = Literal[
    "lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"
]
RankMethod: TypeAlias = Literal["average", "min", "max", "dense", "ordinal", "random"]
PivotAgg: TypeAlias = Literal[
    "first", "sum", "max", "min", "mean", "median", "last", "count"
]
UniqueKeepStrategy: TypeAlias = Literal["first", "last"]
TimeUnit: TypeAlias = Literal["ns", "us", "ms"]

# The following have a Rust enum equivalent with a different name
ToStructStrategy: TypeAlias = Literal[
    "first_non_null", "max_width"
]  # ListToStructWidthStrategy
AsofJoinStrategy: TypeAlias = Literal["backward", "forward"]  # AsofStrategy
InterpolationMethod: TypeAlias = Literal[
    "nearest", "higher", "lower", "midpoint", "linear"
]  # QuantileInterpolOptions

# The following have no equivalent on the Rust side
EpochTimeUnit = Literal["ns", "us", "ms", "s", "d"]
Orientation: TypeAlias = Literal["col", "row"]
TransferEncoding: TypeAlias = Literal["hex", "base64"]
EpochTimeUnit = TimeUnit | Literal["s", "d"]
