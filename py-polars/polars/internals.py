"""Re-export Polars functionality to avoid cyclical imports."""

from polars.dataframe import DataFrame, wrap_df
from polars.expr import Expr, wrap_expr
from polars.lazyframe import LazyFrame, wrap_ldf
from polars.series import Series, wrap_s

__all__ = [
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Series",
    "wrap_df",
    "wrap_expr",
    "wrap_ldf",
    "wrap_s",
]
