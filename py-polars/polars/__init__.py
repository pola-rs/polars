# flake8: noqa
import warnings

try:
    from polars.polars import version
except ImportError as e:  # pragma: no cover

    def version() -> str:
        return ""

    # this is only useful for documentation
    warnings.warn("polars binary missing!")

import polars.testing as testing
from polars.cfg import (  # flake8: noqa. We do not export in __all__
    Config,
    toggle_string_cache,
)
from polars.convert import from_arrow, from_dict, from_dicts, from_pandas, from_records
from polars.datatypes import (
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Object,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)
from polars.internals.expr import Expr
from polars.internals.frame import (  # flake8: noqa # TODO: remove need for wrap_df
    DataFrame,
    wrap_df,
)
from polars.internals.functions import (
    arg_where,
    concat,
    date_range,
    get_dummies,
    repeat,
)
from polars.internals.lazy_frame import LazyFrame
from polars.internals.lazy_functions import _date as date
from polars.internals.lazy_functions import _datetime as datetime
from polars.internals.lazy_functions import (
    all,
    any,
    apply,
    arange,
    argsort_by,
    avg,
    col,
    collect_all,
    concat_list,
    concat_str,
    count,
    cov,
    exclude,
    first,
    fold,
    format,
    groups,
    head,
    last,
    lit,
    map,
    map_binary,
    max,
    mean,
    median,
    min,
    n_unique,
    pearson_corr,
    quantile,
    select,
    spearman_rank_corr,
    std,
    sum,
    tail,
)
from polars.internals.lazy_functions import to_list as list
from polars.internals.lazy_functions import var
from polars.internals.series import (  # flake8: noqa # TODO: remove need for wrap_s
    Series,
    wrap_s,
)
from polars.internals.whenthen import when
from polars.io import (
    read_csv,
    read_ipc,
    read_ipc_schema,
    read_json,
    read_parquet,
    read_sql,
    scan_csv,
    scan_ipc,
    scan_parquet,
)
from polars.string_cache import StringCache

__all__ = [
    "DataFrame",
    "Series",
    "LazyFrame",
    # polars.datatypes
    "DataType",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "Boolean",
    "Utf8",
    "List",
    "Date",
    "Datetime",
    "Time",
    "Object",
    "Categorical",
    # polars.io
    "read_csv",
    "read_parquet",
    "read_json",
    "read_sql",
    "read_ipc",
    "scan_csv",
    "scan_ipc",
    "scan_parquet",
    "read_ipc_schema",
    # polars.stringcache
    "StringCache",
    # polars.config
    "Config",
    # polars.internal.when
    "when",
    # polars.internal.expr
    "Expr",
    # polars.internal.functions
    "arg_where",
    "concat",
    "date_range",
    "get_dummies",
    "repeat",
    # polars.internal.lazy_functions
    "col",
    "count",
    "std",
    "var",
    "max",
    "min",
    "sum",
    "mean",
    "avg",
    "median",
    "n_unique",
    "first",
    "last",
    "head",
    "tail",
    "lit",
    "pearson_corr",
    "spearman_rank_corr",
    "cov",
    "map",
    "apply",
    "map_binary",
    "fold",
    "any",
    "all",
    "groups",
    "quantile",
    "arange",
    "argsort_by",
    "concat_str",
    "concat_list",
    "collect_all",
    "exclude",
    "format",
    "datetime",  # named _datetime, see import above
    "date",  # name _date, see import above
    "list",  # named to_list, see import above
    "select",
    "var",
    # polars.convert
    "from_dict",
    "from_dicts",
    "from_records",
    "from_arrow",
    "from_pandas",
    # testing
    "testing",
]

__version__ = version()

import os

os.environ["POLARS_ALLOW_EXTENSION"] = "true"
