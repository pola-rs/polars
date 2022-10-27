import warnings

try:
    from polars.polars import version
except ImportError:

    def version() -> str:
        return ""

    # this is only useful for documentation
    warnings.warn("polars binary missing!")

from polars.cfg import Config
from polars.convert import (
    from_arrow,
    from_dict,
    from_dicts,
    from_numpy,
    from_pandas,
    from_records,
)
from polars.datatypes import (
    Binary,
    Boolean,
    Categorical,
    DataType,
    Date,
    Datetime,
    Duration,
    Field,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    Null,
    Object,
    PolarsDataType,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Unknown,
    Utf8,
    get_idx_type,
)
from polars.exceptions import (
    ArrowError,
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    NoDataError,
    NotFoundError,
    PanicException,
    SchemaError,
    ShapeError,
)
from polars.internals import BatchedCsvReader

# TODO remove need for wrap_df
from polars.internals.dataframe import wrap_df  # noqa: F401
from polars.internals.dataframe import DataFrame
from polars.internals.expr import Expr
from polars.internals.functions import (
    align_frames,
    concat,
    cut,
    date_range,
    get_dummies,
    ones,
    zeros,
)
from polars.internals.io import read_ipc_schema, read_parquet_schema
from polars.internals.lazy_functions import _date as date
from polars.internals.lazy_functions import _datetime as datetime
from polars.internals.lazy_functions import (
    all,
    any,
    apply,
    arange,
    arg_where,
    argsort_by,
    avg,
    coalesce,
    col,
    collect_all,
    concat_list,
    concat_str,
    count,
    cov,
    cumfold,
    cumsum,
    duration,
    element,
    exclude,
    first,
    fold,
    format,
    groups,
    head,
    last,
    lit,
    map,
    max,
    mean,
    median,
    min,
    n_unique,
    pearson_corr,
    quantile,
    repeat,
    select,
    spearman_rank_corr,
    std,
    struct,
    sum,
    tail,
)
from polars.internals.lazy_functions import to_list as list
from polars.internals.lazy_functions import var
from polars.internals.lazyframe import LazyFrame

# TODO: remove need for wrap_s
from polars.internals.series import wrap_s  # noqa: F401
from polars.internals.series import Series
from polars.internals.sql import SQLContext
from polars.internals.whenthen import when
from polars.io import (
    read_avro,
    read_csv,
    read_csv_batched,
    read_excel,
    read_ipc,
    read_json,
    read_ndjson,
    read_parquet,
    read_sql,
    scan_csv,
    scan_ds,
    scan_ipc,
    scan_ndjson,
    scan_parquet,
)
from polars.show_versions import show_versions
from polars.string_cache import StringCache, toggle_string_cache, using_string_cache
from polars.utils import threadpool_size

__all__ = [
    "exceptions",
    "NotFoundError",
    "ShapeError",
    "SchemaError",
    "ArrowError",
    "BatchedCsvReader",
    "ComputeError",
    "InvalidOperationError",
    "NoDataError",
    "DuplicateError",
    "PanicException",
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
    "Binary",
    "Utf8",
    "List",
    "Date",
    "Datetime",
    "Time",
    "Duration",
    "Object",
    "Categorical",
    "Field",
    "Struct",
    "Null",
    "Unknown",
    "PolarsDataType",
    "get_idx_type",
    # polars.io
    "read_csv",
    "read_csv_batched",
    "read_excel",
    "read_parquet",
    "read_json",
    "read_ndjson",
    "read_sql",
    "read_ipc",
    "scan_csv",
    "scan_ipc",
    "scan_ds",
    "scan_parquet",
    "scan_ndjson",
    "read_ipc_schema",
    "read_parquet_schema",
    "read_avro",
    # polars.stringcache
    "StringCache",
    "toggle_string_cache",
    "using_string_cache",
    # polars.config
    "Config",
    # polars.internals.whenthen
    "when",
    # polars.internals.expr
    "Expr",
    # polars.internals.functions
    "align_frames",
    "arg_where",
    "concat",
    "date_range",
    "get_dummies",
    "ones",
    "repeat",
    "element",
    "cut",
    "zeros",
    # polars.internals.lazy_functions
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
    "fold",
    "cumfold",
    "cumsum",
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
    "struct",
    "duration",
    "coalesce",
    # polars.convert
    "from_dict",
    "from_dicts",
    "from_records",
    "from_numpy",
    "from_arrow",
    "from_pandas",
    # testing
    "threadpool_size",
    # version
    "show_versions",
    "SQLContext",
]

__version__ = version()

import os

os.environ["POLARS_ALLOW_EXTENSION"] = "true"
