from __future__ import annotations

import sys
from typing import Union

from polars import internals as pli

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

IntoExpr = Union[int, float, str, "pli.Expr", "pli.Series"]

ClosedWindow = Literal["left", "right", "both", "none"]
FillStrategy = Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
InterpolationMethod = Literal["nearest", "higher", "lower", "midpoint", "linear"]
