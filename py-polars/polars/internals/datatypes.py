from __future__ import annotations

from typing import Union

from polars import internals as pli

IntoExpr = Union[int, float, str, "pli.Expr", "pli.Series"]
