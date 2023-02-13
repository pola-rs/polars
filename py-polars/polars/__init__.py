import os
import warnings

try:
    from polars.polars import version
except ImportError:

    def version() -> str:
        return ""

    # this is only useful for documentation
    warnings.warn("polars binary missing!")

from polars import api
from polars.build_info import build_info
from polars.cfg import Config
from polars.convert import (
    from_arrow,
    from_dataframe,
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
    ColumnNotFoundError,
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    NoDataError,
    PanicException,
    SchemaError,
    SchemaFieldNotFoundError,
    ShapeError,
    StructFieldNotFoundError,
)
from polars.internals import BatchedCsvReader

# TODO remove need for wrap_df
from polars.internals.dataframe import (
    DataFrame,
    wrap_df,  # noqa: F401
)
from polars.internals.expr.expr import Expr
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
    arg_sort_by,
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
    cumreduce,
    cumsum,
    duration,
    element,
    exclude,
    first,
    fold,
    format,
    from_epoch,
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
    reduce,
    repeat,
    select,
    spearman_rank_corr,
    std,
    struct,
    sum,
    tail,
    var,
)
from polars.internals.lazy_functions import to_list as list
from polars.internals.lazyframe import LazyFrame

# TODO: remove need for wrap_s
from polars.internals.series import wrap_s  # noqa: F401
from polars.internals.series.series import Series
from polars.internals.sql import SQLContext
from polars.internals.whenthen import when
from polars.io import (
    read_avro,
    read_csv,
    read_csv_batched,
    read_delta,
    read_excel,
    read_ipc,
    read_json,
    read_ndjson,
    read_parquet,
    read_sql,
    scan_csv,
    scan_delta,
    scan_ds,
    scan_ipc,
    scan_ndjson,
    scan_parquet,
)
from polars.show_versions import show_versions
from polars.string_cache import StringCache, toggle_string_cache, using_string_cache
from polars.utils import threadpool_size

__all__ = [
    "api",
    "exceptions",
    # exceptions/errors
    "ArrowError",
    "ColumnNotFoundError",
    "ComputeError",
    "DuplicateError",
    "InvalidOperationError",
    "NoDataError",
    "PanicException",
    "SchemaError",
    "SchemaFieldNotFoundError",
    "ShapeError",
    "StructFieldNotFoundError",
    # core classes
    "BatchedCsvReader",
    "DataFrame",
    "Expr",
    "LazyFrame",
    "Series",
    # polars.datatypes
    "Binary",
    "Boolean",
    "Categorical",
    "DataType",
    "Date",
    "Datetime",
    "Duration",
    "Field",
    "Float32",
    "Float64",
    "Int16",
    "Int32",
    "Int64",
    "Int8",
    "List",
    "Null",
    "Object",
    "PolarsDataType",
    "Struct",
    "Time",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt8",
    "Unknown",
    "Utf8",
    "get_idx_type",
    # polars.io
    "read_avro",
    "read_csv",
    "read_csv_batched",
    "read_delta",
    "read_excel",
    "read_ipc",
    "read_ipc_schema",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "read_parquet_schema",
    "read_sql",
    "scan_csv",
    "scan_delta",
    "scan_ds",
    "scan_ipc",
    "scan_ndjson",
    "scan_parquet",
    # polars.stringcache
    "StringCache",
    "toggle_string_cache",
    "using_string_cache",
    # polars.config
    "Config",
    # polars.internals.whenthen
    "when",
    # polars.internals.functions
    "align_frames",
    "arg_where",
    "concat",
    "cut",
    "date_range",
    "element",
    "get_dummies",
    "ones",
    "repeat",
    "zeros",
    # polars.internals.lazy_functions
    "all",
    "any",
    "apply",
    "arange",
    "arg_sort_by",
    "argsort_by",
    "avg",
    "coalesce",
    "col",
    "collect_all",
    "concat_list",
    "concat_str",
    "count",
    "cov",
    "cumfold",
    "cumreduce",
    "cumsum",
    "date",  # name _date, see import above
    "datetime",  # named _datetime, see import above
    "duration",
    "exclude",
    "first",
    "fold",
    "format",
    "from_epoch",
    "groups",
    "head",
    "last",
    "list",  # named to_list, see import above
    "lit",
    "map",
    "max",
    "mean",
    "median",
    "min",
    "n_unique",
    "pearson_corr",
    "quantile",
    "reduce",
    "select",
    "spearman_rank_corr",
    "std",
    "struct",
    "sum",
    "tail",
    "var",
    "var",
    # polars.convert
    "from_arrow",
    "from_dataframe",
    "from_dict",
    "from_dicts",
    "from_numpy",
    "from_pandas",
    "from_records",
    # testing
    "threadpool_size",
    # version
    "SQLContext",
    "build_info",
    "show_versions",
]

__version__ = version()

os.environ["POLARS_ALLOW_EXTENSION"] = "true"
