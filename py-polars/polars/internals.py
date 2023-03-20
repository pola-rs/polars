"""Re-export Polars functionality to avoid cyclical imports."""

from polars.dataframe import DataFrame
from polars.expr import Expr
from polars.lazyframe import LazyFrame
from polars.series import Series
from polars.utils._wrap import wrap_df, wrap_expr, wrap_ldf, wrap_s

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
