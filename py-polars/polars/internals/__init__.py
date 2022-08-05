"""
The modules within `polars.internals` are interdependent. To prevent cyclical imports,
they all import from each other via this __init__ file using
`import polars.internals as pli`. The imports below are being shared across this module.
"""
from .anonymous_scan import (
    _deser_and_exec,
    _scan_ds,
    _scan_ipc_fsspec,
    _scan_parquet_fsspec,
)
from .datatypes import IntoExpr
from .expr import Expr, expr_to_lit_or_expr, selection_to_pyexpr_list, wrap_expr
from .frame import DataFrame, wrap_df
from .functions import concat, date_range
from .io import _is_local_file, _prepare_file_arg, read_ipc_schema, read_parquet_schema
from .lazy_frame import LazyFrame, wrap_ldf
from .lazy_functions import (
    all,
    arg_where,
    argsort_by,
    col,
    concat_list,
    element,
    format,
    lit,
    select,
)
from .series import Series, wrap_s
from .whenthen import when  # used in expr.clip()

__all__ = [
    "DataFrame",
    "Expr",
    "IntoExpr",
    "LazyFrame",
    "Series",
    "all",
    "arg_where",
    "argsort_by",
    "col",
    "concat",
    "concat_list",
    "date_range",
    "element",
    "expr_to_lit_or_expr",
    "format",
    "lit",
    "read_ipc_schema",
    "read_parquet_schema",
    "select",
    "selection_to_pyexpr_list",
    "when",
    "wrap_df",
    "wrap_expr",
    "wrap_ldf",
    "wrap_s",
    "_deser_and_exec",
    "_is_local_file",
    "_prepare_file_arg",
    "_scan_ds",
    "_scan_ipc_fsspec",
    "_scan_parquet_fsspec",
]
