"""
Re-export Polars functionality to avoid cyclical imports.

If you run into cyclical imports, export functionality in this module, then import this
module using `from polars import internals as pli`.
"""
from polars.dataframe import DataFrame, wrap_df
from polars.expr import (
    Expr,
    expr_to_lit_or_expr,
    selection_to_pyexpr_list,
    wrap_expr,
)
from polars.functions.eager import concat, date_range
from polars.functions.lazy import (
    all,
    arange,
    arg_sort_by,
    arg_where,
    argsort_by,
    coalesce,
    col,
    collect_all,
    concat_list,
    count,
    element,
    format,
    from_epoch,
    lit,
    select,
    struct,
)
from polars.functions.whenthen import WhenThen, WhenThenThen, when
from polars.lazyframe import LazyFrame, wrap_ldf
from polars.series import Series, wrap_s

__all__ = [
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Series",
    "all",
    "arange",
    "arg_where",
    "arg_sort_by",
    "argsort_by",
    "coalesce",
    "col",
    "collect_all",
    "concat",
    "concat_list",
    "count",
    "date_range",
    "element",
    "expr_to_lit_or_expr",
    "format",
    "from_epoch",
    "lit",
    "select",
    "selection_to_pyexpr_list",
    "struct",
    "when",
    "wrap_df",
    "wrap_expr",
    "wrap_ldf",
    "wrap_s",
    "WhenThen",
    "WhenThenThen",
]
