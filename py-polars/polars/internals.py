"""
Re-export Polars functionality to avoid cyclical imports.

If you run into cyclical imports, export functionality in this module, then import this
module using `from polars import internals as pli`.
"""
from polars.dataframe import DataFrame, wrap_df
from polars.expr import Expr, expr_to_lit_or_expr, selection_to_pyexpr_list, wrap_expr
from polars.lazyframe import LazyFrame, wrap_ldf
from polars.series import Series, wrap_s

__all__ = [
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Series",
    "expr_to_lit_or_expr",
    "selection_to_pyexpr_list",
    "wrap_df",
    "wrap_expr",
    "wrap_ldf",
    "wrap_s",
]
