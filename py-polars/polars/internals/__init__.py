"""
Core Polars functionality.

The modules within `polars.internals` are interdependent. To prevent cyclical imports,
they all import from each other via this __init__ file using
`import polars.internals as pli`. The imports below are being shared across this module.
"""
from polars.internals.anonymous_scan import (
    _deser_and_exec,
    _scan_ds,
    _scan_ipc_fsspec,
    _scan_parquet_fsspec,
)
from polars.internals.batched import BatchedCsvReader
from polars.internals.dataframe import DataFrame, wrap_df
from polars.internals.expr import (
    Expr,
    expr_to_lit_or_expr,
    selection_to_pyexpr_list,
    wrap_expr,
)
from polars.internals.functions import concat, date_range
from polars.internals.io import (
    _is_local_file,
    _prepare_file_arg,
    _update_columns,
    read_ipc_schema,
    read_parquet_schema,
)
from polars.internals.lazy_functions import (
    all,
    arange,
    arg_where,
    argsort_by,
    col,
    concat_list,
    count,
    element,
    format,
    from_epoch,
    lit,
    select,
    struct,
)
from polars.internals.lazyframe import LazyFrame, wrap_ldf
from polars.internals.series import Series, wrap_s
from polars.internals.whenthen import WhenThen, WhenThenThen, when

__all__ = [
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Series",
    "all",
    "arange",
    "arg_where",
    "argsort_by",
    "BatchedCsvReader",
    "col",
    "concat",
    "concat_list",
    "count",
    "date_range",
    "element",
    "expr_to_lit_or_expr",
    "format",
    "from_epoch",
    "lit",
    "read_ipc_schema",
    "read_parquet_schema",
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
    "_deser_and_exec",
    "_is_local_file",
    "_prepare_file_arg",
    "_scan_ds",
    "_scan_ipc_fsspec",
    "_scan_parquet_fsspec",
    "_update_columns",
]
