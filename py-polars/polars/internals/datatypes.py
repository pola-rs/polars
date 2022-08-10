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

# String literal types
ClosedWindow: TypeAlias = Literal["left", "right", "both", "none"]
FillStrategy: TypeAlias = Literal[
    "forward", "backward", "min", "max", "mean", "zero", "one"
]
InterpolationMethod: TypeAlias = Literal[
    "nearest", "higher", "lower", "midpoint", "linear"
]
Orientation: TypeAlias = Literal["col", "row"]
