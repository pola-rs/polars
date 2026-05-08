from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polars._utils.unstable import unstable

if TYPE_CHECKING:
    import polars._reexport as pl


@unstable()
@dataclass(frozen=True, slots=True)
class AsofJoinPair:
    left_on: str | pl.Expr
    right_on: str | pl.Expr
    suffix: str | None = None
