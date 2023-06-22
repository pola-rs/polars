#![feature(vec_into_raw_parts)]
#![allow(clippy::nonstandard_macro_braces)] // needed because clippy does not understand proc macro of pyo3
#![allow(clippy::transmute_undefined_repr)]
extern crate polars as polars_rs;

#[cfg(feature = "build_info")]
#[macro_use]
extern crate pyo3_built;

#[cfg(feature = "build_info")]
#[allow(dead_code)]
mod build {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}
pub mod apply;
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
pub mod lazyframe;
pub mod lazygroupby;
#[cfg(feature = "object")]
mod object;
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
pub use object::__register_startup_deps;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::error::{
    ArrowErrorException, ColumnNotFoundError, ComputeError, DuplicateError, InvalidOperationError,
    NoDataError, PyPolarsErr, SchemaError, SchemaFieldNotFoundError, StructFieldNotFoundError,
};
use crate::expr::PyExpr;
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
    #[cfg(feature = "csv")]
    m.add_class::<batched_csv::PyBatchedCsv>().unwrap();
    #[cfg(feature = "sql")]
    m.add_class::<sql::PySQLContext>().unwrap();

    // Functions - eager
    m.add_wrapped(wrap_pyfunction!(functions::eager::concat_df))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::concat_series))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::date_range_eager))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::diag_concat_df))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::hor_concat_df))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::eager::time_range_eager))
        .unwrap();

    // Functions - lazy
    m.add_wrapped(wrap_pyfunction!(functions::lazy::arange))
        .unwrap();
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
    m.add_wrapped(wrap_pyfunction!(functions::lazy::date_range_lazy))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::datetime))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::diag_concat_lf))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::concat_expr))
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
    m.add_wrapped(wrap_pyfunction!(functions::lazy::max_exprs))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::min_exprs))
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
    m.add_wrapped(wrap_pyfunction!(functions::lazy::sum_exprs))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lazy::time_range_lazy))
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
    m.add_wrapped(wrap_pyfunction!(functions::meta::enable_string_cache))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::using_string_cache))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::set_float_fmt))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::meta::get_float_fmt))
        .unwrap();

    // Functions - misc
    m.add_wrapped(wrap_pyfunction!(functions::misc::dtype_str_repr))
        .unwrap();
    #[cfg(feature = "object")]
    m.add_wrapped(wrap_pyfunction!(__register_startup_deps))
        .unwrap();

    // Exceptions
    m.add("ArrowError", py.get_type::<ArrowErrorException>())
        .unwrap();
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
