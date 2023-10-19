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
pub mod arrow_interop;
#[cfg(feature = "csv")]
mod batched_csv;
pub mod conversion;
pub mod dataframe;
pub mod datatypes;
pub mod error;
pub mod expr;
pub mod file;
pub mod functions;
pub(crate) mod gil_once_cell;
pub mod lazyframe;
pub mod lazygroupby;
pub mod map;
#[cfg(feature = "object")]
mod object;
#[cfg(feature = "object")]
mod on_startup;
pub mod prelude;
pub(crate) mod py_modules;
pub mod series;
#[cfg(feature = "sql")]
mod sql;
pub mod utils;

#[cfg(all(target_os = "linux", not(use_mimalloc)))]
use jemallocator::Jemalloc;
#[cfg(any(not(target_os = "linux"), use_mimalloc))]
use mimalloc::MiMalloc;
#[cfg(feature = "object")]
pub use on_startup::__register_startup_deps;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::error::{
    ColumnNotFoundError, ComputeError, DuplicateError, InvalidOperationError, NoDataError,
    OutOfBoundsError, PyPolarsErr, SchemaError, SchemaFieldNotFoundError, StructFieldNotFoundError,
};
use crate::expr::PyExpr;
use crate::functions::string_cache::PyStringCacheHolder;
use crate::lazyframe::PyLazyFrame;
use crate::lazygroupby::PyLazyGroupBy;
use crate::series::PySeries;

#[global_allocator]
#[cfg(all(target_os = "linux", not(use_mimalloc)))]
static ALLOC: Jemalloc = Jemalloc;

#[global_allocator]
#[cfg(any(not(target_os = "linux"), use_mimalloc))]
static ALLOC: MiMalloc = MiMalloc;

#[pymodule]
fn polars(py: Python, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<PyExpr>().unwrap();
    m.add_class::<PyStringCacheHolder>().unwrap();
    #[cfg(feature = "csv")]
    m.add_class::<batched_csv::PyBatchedCsv>().unwrap();
    #[cfg(feature = "sql")]
    m.add_class::<sql::PySQLContext>().unwrap();

    // Functions - eager
    m.add_wrapped(wrap_pyfunction!(functions::eager::concat_df))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::concat_series))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::concat_df_diagonal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::concat_df_horizontal))
        .unwrap();

    // Functions - range
    m.add_wrapped(wrap_pyfunction!(functions::range::int_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::range::int_ranges))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::range::date_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::range::date_ranges))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::range::datetime_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::range::datetime_ranges))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::range::time_range))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::range::time_ranges))
        .unwrap();

    // Functions - aggregation
    m.add_wrapped(wrap_pyfunction!(functions::aggregation::all_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::aggregation::any_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::aggregation::max_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::aggregation::min_horizontal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::aggregation::sum_horizontal))
        .unwrap();

    // Functions - lazy
    m.add_wrapped(wrap_pyfunction!(functions::lazy::arg_sort_by))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::arg_where))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::as_struct))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::coalesce))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::col))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::collect_all))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::collect_all_with_callback))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::cols))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_lf))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_list))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_str))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::count))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::cov))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::cumfold))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::cumreduce))
        .unwrap();
    #[cfg(feature = "trigonometry")]
    m.add_wrapped(wrap_pyfunction!(functions::lazy::arctan2))
        .unwrap();
    #[cfg(feature = "trigonometry")]
    m.add_wrapped(wrap_pyfunction!(functions::lazy::arctan2d))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::datetime))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_expr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_lf_diagonal))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::dtype_cols))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::duration))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::first))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::fold))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::last))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::lit))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::map_mul))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::pearson_corr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::rolling_corr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::rolling_cov))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::reduce))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::repeat))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::spearman_rank_corr))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::whenthen::when))
        .unwrap();

    #[cfg(feature = "sql")]
    m.add_wrapped(wrap_pyfunction!(functions::lazy::sql_expr))
        .unwrap();

    // Functions - I/O
    #[cfg(feature = "ipc")]
    m.add_wrapped(wrap_pyfunction!(functions::io::read_ipc_schema))
        .unwrap();
    #[cfg(feature = "parquet")]
    m.add_wrapped(wrap_pyfunction!(functions::io::read_parquet_schema))
        .unwrap();

    // Functions - meta
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_polars_version))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_index_type))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::threadpool_size))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(
        functions::string_cache::enable_string_cache
    ))
    .unwrap();
    m.add_wrapped(wrap_pyfunction!(
        functions::string_cache::disable_string_cache
    ))
    .unwrap();
    m.add_wrapped(wrap_pyfunction!(
        functions::string_cache::using_string_cache
    ))
    .unwrap();

    // Numeric formatting
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_digit_group_separator))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::set_digit_group_separator))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_float_fmt))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_float_precision))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_decimal_separator))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_digit_group_size))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_trim_decimal_zeros))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::set_float_fmt))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::set_float_precision))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::set_decimal_separator))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::set_digit_group_size))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::set_trim_decimal_zeros))
        .unwrap();

    // Functions - misc
    m.add_wrapped(wrap_pyfunction!(functions::misc::dtype_str_repr))
        .unwrap();
    #[cfg(feature = "object")]
    m.add_wrapped(wrap_pyfunction!(__register_startup_deps))
        .unwrap();

    // Functions - random
    m.add_wrapped(wrap_pyfunction!(functions::random::set_random_seed))
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
