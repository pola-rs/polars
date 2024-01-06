#![feature(vec_into_raw_parts)]
#![allow(clippy::nonstandard_macro_braces)] // Needed because clippy does not understand proc macro of PyO3
#![allow(clippy::transmute_undefined_repr)]
#![allow(clippy::too_many_arguments)] // Python functions can have many arguments due to default arguments
extern crate polars as polars_rs;

#[cfg(feature = "build_info")]
#[macro_use]
extern crate pyo3_built;

#[cfg(feature = "build_info")]
#[allow(dead_code)]
mod build {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

mod arrow_interop;
#[cfg(feature = "csv")]
mod batched_csv;
mod conversion;
mod dataframe;
mod datatypes;
mod error;
mod expr;
mod file;
mod functions;
mod gil_once_cell;
mod lazyframe;
mod lazygroupby;
mod map;
#[cfg(feature = "object")]
mod object;
#[cfg(feature = "object")]
mod on_startup;
mod prelude;
mod py_modules;
mod series;
#[cfg(feature = "sql")]
mod sql;
mod utils;

#[cfg(all(target_family = "unix", not(use_mimalloc)))]
use jemallocator::Jemalloc;
#[cfg(any(not(target_family = "unix"), use_mimalloc))]
use mimalloc::MiMalloc;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[cfg(feature = "csv")]
use crate::batched_csv::PyBatchedCsv;
use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::error::{
    CategoricalRemappingWarning, ColumnNotFoundError, ComputeError, DuplicateError,
    InvalidOperationError, NoDataError, OutOfBoundsError, PyPolarsErr, SchemaError,
    SchemaFieldNotFoundError, StructFieldNotFoundError,
};
use crate::expr::PyExpr;
use crate::functions::PyStringCacheHolder;
use crate::lazyframe::{PyInProcessQuery, PyLazyFrame};
use crate::lazygroupby::PyLazyGroupBy;
use crate::series::PySeries;
#[cfg(feature = "sql")]
use crate::sql::PySQLContext;

#[global_allocator]
#[cfg(all(target_family = "unix", not(use_mimalloc)))]
static ALLOC: Jemalloc = Jemalloc;

#[global_allocator]
#[cfg(any(not(target_family = "unix"), use_mimalloc))]
static ALLOC: MiMalloc = MiMalloc;

#[pymodule]
fn polars(py: Python, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyInProcessQuery>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<PyExpr>().unwrap();
    m.add_class::<PyStringCacheHolder>().unwrap();
    #[cfg(feature = "csv")]
    m.add_class::<PyBatchedCsv>().unwrap();
    #[cfg(feature = "sql")]
    m.add_class::<PySQLContext>().unwrap();

    // Functions - eager
    m.add_wrapped(wrap_pyfunction!(functions::concat_df))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_series))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_df_diagonal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_df_horizontal))
        .unwrap();

    // Functions - range
    m.add_wrapped(wrap_pyfunction!(functions::int_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::int_ranges))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::date_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::date_ranges))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::datetime_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::datetime_ranges))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::time_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::time_ranges))
        .unwrap();

    // Functions - aggregation
    m.add_wrapped(wrap_pyfunction!(functions::all_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::any_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::max_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::min_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::sum_horizontal))
        .unwrap();

    // Functions - lazy
    m.add_wrapped(wrap_pyfunction!(functions::arg_sort_by))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::arg_where))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::as_struct))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::coalesce))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::col)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::collect_all))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::collect_all_with_callback))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::cols)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_lf))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_list))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_str))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::count)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::cov)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::cum_count))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::cum_fold))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::cum_reduce))
        .unwrap();
    #[cfg(feature = "trigonometry")]
    m.add_wrapped(wrap_pyfunction!(functions::arctan2)).unwrap();
    #[cfg(feature = "trigonometry")]
    m.add_wrapped(wrap_pyfunction!(functions::arctan2d))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::datetime))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_expr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_lf_diagonal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_lf_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::dtype_cols))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::duration))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::first)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::fold)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::last)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lit)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::map_mul)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::pearson_corr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::rolling_corr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::rolling_cov))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::reduce)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::repeat)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::spearman_rank_corr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::when)).unwrap();

    #[cfg(feature = "sql")]
    m.add_wrapped(wrap_pyfunction!(functions::sql_expr))
        .unwrap();

    // Functions - I/O
    #[cfg(feature = "ipc")]
    m.add_wrapped(wrap_pyfunction!(functions::read_ipc_schema))
        .unwrap();
    #[cfg(feature = "parquet")]
    m.add_wrapped(wrap_pyfunction!(functions::read_parquet_schema))
        .unwrap();

    // Functions - meta
    m.add_wrapped(wrap_pyfunction!(functions::get_polars_version))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::get_index_type))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::threadpool_size))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::enable_string_cache))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::disable_string_cache))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::using_string_cache))
        .unwrap();

    // Numeric formatting
    m.add_wrapped(wrap_pyfunction!(functions::get_thousands_separator))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::set_thousands_separator))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::get_float_fmt))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::get_float_precision))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::get_decimal_separator))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::get_trim_decimal_zeros))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::set_float_fmt))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::set_float_precision))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::set_decimal_separator))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::set_trim_decimal_zeros))
        .unwrap();

    // Functions - misc
    m.add_wrapped(wrap_pyfunction!(functions::dtype_str_repr))
        .unwrap();
    #[cfg(feature = "object")]
    m.add_wrapped(wrap_pyfunction!(on_startup::__register_startup_deps))
        .unwrap();

    // Functions - random
    m.add_wrapped(wrap_pyfunction!(functions::set_random_seed))
        .unwrap();

    // Exceptions
    m.add("ColumnNotFoundError", py.get_type::<ColumnNotFoundError>())
        .unwrap();
    m.add("ComputeError", py.get_type::<ComputeError>())
        .unwrap();
    m.add("DuplicateError", py.get_type::<DuplicateError>())
        .unwrap();
    m.add(
        "InvalidOperationError",
        py.get_type::<InvalidOperationError>(),
    )
    .unwrap();
    m.add("NoDataError", py.get_type::<NoDataError>()).unwrap();
    m.add(
        "CategoricalRemappingWarning",
        py.get_type::<CategoricalRemappingWarning>(),
    )
    .unwrap();
    m.add("OutOfBoundsError", py.get_type::<OutOfBoundsError>())
        .unwrap();
    m.add("PolarsPanicError", py.get_type::<PanicException>())
        .unwrap();
    m.add("SchemaError", py.get_type::<SchemaError>()).unwrap();
    m.add(
        "SchemaFieldNotFoundError",
        py.get_type::<SchemaFieldNotFoundError>(),
    )
    .unwrap();
    m.add("ShapeError", py.get_type::<crate::error::ShapeError>())
        .unwrap();
    m.add(
        "StringCacheMismatchError",
        py.get_type::<crate::error::StringCacheMismatchError>(),
    )
    .unwrap();
    m.add(
        "StructFieldNotFoundError",
        py.get_type::<StructFieldNotFoundError>(),
    )
    .unwrap();

    // Build info
    #[cfg(feature = "build_info")]
    m.add(
        "_build_info_",
        pyo3_built!(py, build, "build", "time", "deps", "features", "host", "target", "git"),
    )?;

    Ok(())
}
