# flake8: noqa
import warnings

try:
    from polars.polars import version
except ImportError as e:
    version = lambda: ""
    # this is only useful for documentation
    warnings.warn("polars binary missing!")

from polars.internals.frame import DataFrame
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
    to_list,
    var,
)
from polars.internals.series import (  # TODO: this top-level import fixes a number of tests, but we should not want this import here
    Series,
    wrap_s,
)
from polars.internals.whenthen import when

# TODO: remove wildcard imports
from .convert import *
from .datatypes import *
from .io import *
from .string_cache import StringCache

__version__ = version()
