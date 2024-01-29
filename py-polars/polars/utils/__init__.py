"""
Utility functions.

Functions that are part of the public API are re-exported here.
"""
from polars.utils._scan import _execute_from_rust
from polars.utils.build_info import build_info
from polars.utils.convert import (
    _date_to_pl_date,
    _datetime_for_any_value,
    _datetime_for_any_value_windows,
    _time_to_pl_time,
    _timedelta_to_pl_timedelta,
    _to_python_date,
    _to_python_datetime,
    _to_python_decimal,
    _to_python_time,
    _to_python_timedelta,
)
from polars.utils.meta import get_index_type, threadpool_size
from polars.utils.show_versions import show_versions
from polars.utils.various import NoDefault, _polars_warn, is_column, no_default

__all__ = [
    "NoDefault",
    "build_info",
    "get_index_type",
    "is_column",
    "no_default",
    "show_versions",
    "threadpool_size",
    # Required for Rust bindings
    "_date_to_pl_date",
    "_datetime_for_any_value",
    "_datetime_for_any_value_windows",
    "_execute_from_rust",
    "_polars_warn",
    "_time_to_pl_time",
    "_timedelta_to_pl_timedelta",
    "_to_python_date",
    "_to_python_datetime",
    "_to_python_decimal",
    "_to_python_time",
    "_to_python_timedelta",
]
