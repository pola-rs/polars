from __future__ import annotations

import sys
from typing import Union

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

# User-facing string literal types
ClosedWindow: TypeAlias = Literal["left", "right", "both", "none"]
FillStrategy: TypeAlias = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
NullStrategy: TypeAlias = Literal["ignore", "propagate"]

InterpolationMethod: TypeAlias = Literal[
    "nearest", "higher", "lower", "midpoint", "linear"
]
Orientation: TypeAlias = Literal["col", "row"]

FileEncoding: TypeAlias = Literal["utf8", "utf8-lossy"]
TransferEncoding: TypeAlias = Literal["hex", "base64"]

ParallelStrategy: TypeAlias = Literal["auto", "columns", "row_groups", "none"]
ToStructStrategy: TypeAlias = Literal["first_non_null", "max_width"]
AsofJoinStrategy: TypeAlias = Literal["backward", "forward"]

AvroCompression: TypeAlias = Literal["uncompressed", "snappy", "deflate"]
IpcCompression: TypeAlias = Literal["uncompressed", "lz4", "zstd"]
ParquetCompression = Literal[
    "lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"
]

# Other literal types
ComparisonOperator: TypeAlias = Literal["eq", "neq", "gt", "lt", "gt_eq", "lt_eq"]
