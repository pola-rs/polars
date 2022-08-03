from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from polars import internals as pli

IntoExpr = Union[int, float, str, "pli.Expr", "pli.Series"]
