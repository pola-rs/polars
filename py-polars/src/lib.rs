#![allow(clippy::nonstandard_macro_braces)] // Needed because clippy does not understand proc macro of PyO3
#![allow(clippy::transmute_undefined_repr)]
#![allow(non_local_definitions)]
#![allow(clippy::too_many_arguments)] // Python functions can have many arguments due to default arguments

#[cfg(feature = "build_info")]
#[allow(dead_code)]
mod build {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

mod allocator;
#[cfg(debug_assertions)]
mod memory;

use allocator::create_allocator_capsule;
#[cfg(feature = "csv")]
use polars_python::batched_csv::PyBatchedCsv;
#[cfg(feature = "polars_cloud")]
use polars_python::cloud;
use polars_python::dataframe::PyDataFrame;
use polars_python::expr::PyExpr;
use polars_python::functions::PyStringCacheHolder;
use polars_python::lazyframe::{PyInProcessQuery, PyLazyFrame};
use polars_python::lazygroupby::PyLazyGroupBy;
use polars_python::series::PySeries;
#[cfg(feature = "sql")]
use polars_python::sql::PySQLContext;
use polars_python::{exceptions, functions};
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule};

#[pymodule]
fn _ir_nodes(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use polars_python::lazyframe::visitor::nodes::*;
    m.add_class::<PythonScan>().unwrap();
    m.add_class::<Slice>().unwrap();
    m.add_class::<Filter>().unwrap();
    m.add_class::<Scan>().unwrap();
    m.add_class::<DataFrameScan>().unwrap();
    m.add_class::<SimpleProjection>().unwrap();
    m.add_class::<Select>().unwrap();
    m.add_class::<Sort>().unwrap();
    m.add_class::<Cache>().unwrap();
    m.add_class::<GroupBy>().unwrap();
    m.add_class::<Join>().unwrap();
    m.add_class::<HStack>().unwrap();
    m.add_class::<Reduce>().unwrap();
    m.add_class::<Distinct>().unwrap();
    m.add_class::<MapFunction>().unwrap();
    m.add_class::<Union>().unwrap();
    m.add_class::<HConcat>().unwrap();
    m.add_class::<ExtContext>().unwrap();
    m.add_class::<Sink>().unwrap();
    Ok(())
}

#[pymodule]
fn _expr_nodes(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use polars_python::lazyframe::visit::PyExprIR;
    use polars_python::lazyframe::visitor::expr_nodes::*;
    // Expressions
    m.add_class::<PyExprIR>().unwrap();
    m.add_class::<Alias>().unwrap();
    m.add_class::<Column>().unwrap();
    m.add_class::<Literal>().unwrap();
    m.add_class::<BinaryExpr>().unwrap();
    m.add_class::<Cast>().unwrap();
    m.add_class::<Sort>().unwrap();
    m.add_class::<Gather>().unwrap();
    m.add_class::<Filter>().unwrap();
    m.add_class::<SortBy>().unwrap();
    m.add_class::<Agg>().unwrap();
    m.add_class::<Ternary>().unwrap();
    m.add_class::<Function>().unwrap();
    m.add_class::<Slice>().unwrap();
    m.add_class::<Len>().unwrap();
    m.add_class::<Window>().unwrap();
    m.add_class::<PyOperator>().unwrap();
    m.add_class::<PyStringFunction>().unwrap();
    m.add_class::<PyBooleanFunction>().unwrap();
    m.add_class::<PyTemporalFunction>().unwrap();
    // Options
    m.add_class::<PyWindowMapping>().unwrap();
    m.add_class::<PyRollingGroupOptions>().unwrap();
    m.add_class::<PyGroupbyOptions>().unwrap();
    Ok(())
}

#[pymodule]
fn polars(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
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

    // Submodules
    // LogicalPlan objects
    m.add_wrapped(wrap_pymodule!(_ir_nodes))?;
    // Expr objects
    m.add_wrapped(wrap_pymodule!(_expr_nodes))?;

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
    m.add_wrapped(wrap_pyfunction!(functions::eager_int_range))
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

    // Functions - business
    m.add_wrapped(wrap_pyfunction!(functions::business_day_count))
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
    m.add_wrapped(wrap_pyfunction!(functions::mean_horizontal))
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
    m.add_wrapped(wrap_pyfunction!(functions::field)).unwrap();
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
    m.add_wrapped(wrap_pyfunction!(functions::len)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::cov)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::cum_fold))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::cum_reduce))
        .unwrap();
    #[cfg(feature = "trigonometry")]
    m.add_wrapped(wrap_pyfunction!(functions::arctan2)).unwrap();
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
    m.add_wrapped(wrap_pyfunction!(functions::index_cols))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::duration))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::first)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::fold)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::last)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lit)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::map_mul)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::nth)).unwrap();
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
    #[cfg(feature = "clipboard")]
    m.add_wrapped(wrap_pyfunction!(functions::read_clipboard_string))
        .unwrap();
    #[cfg(feature = "clipboard")]
    m.add_wrapped(wrap_pyfunction!(functions::write_clipboard_string))
        .unwrap();

    // Functions - meta
    m.add_wrapped(wrap_pyfunction!(functions::get_index_type))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::thread_pool_size))
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
    m.add_wrapped(wrap_pyfunction!(functions::__register_startup_deps))
        .unwrap();

    // Functions - random
    m.add_wrapped(wrap_pyfunction!(functions::set_random_seed))
        .unwrap();

    // Exceptions - Errors
    m.add(
        "PolarsError",
        py.get_type_bound::<exceptions::PolarsError>(),
    )
    .unwrap();
    m.add(
        "ColumnNotFoundError",
        py.get_type_bound::<exceptions::ColumnNotFoundError>(),
    )
    .unwrap();
    m.add(
        "ComputeError",
        py.get_type_bound::<exceptions::ComputeError>(),
    )
    .unwrap();
    m.add(
        "DuplicateError",
        py.get_type_bound::<exceptions::DuplicateError>(),
    )
    .unwrap();
    m.add(
        "InvalidOperationError",
        py.get_type_bound::<exceptions::InvalidOperationError>(),
    )
    .unwrap();
    m.add(
        "NoDataError",
        py.get_type_bound::<exceptions::NoDataError>(),
    )
    .unwrap();
    m.add(
        "OutOfBoundsError",
        py.get_type_bound::<exceptions::OutOfBoundsError>(),
    )
    .unwrap();
    m.add(
        "SQLInterfaceError",
        py.get_type_bound::<exceptions::SQLInterfaceError>(),
    )
    .unwrap();
    m.add(
        "SQLSyntaxError",
        py.get_type_bound::<exceptions::SQLSyntaxError>(),
    )
    .unwrap();
    m.add(
        "SchemaError",
        py.get_type_bound::<exceptions::SchemaError>(),
    )
    .unwrap();
    m.add(
        "SchemaFieldNotFoundError",
        py.get_type_bound::<exceptions::SchemaFieldNotFoundError>(),
    )
    .unwrap();
    m.add("ShapeError", py.get_type_bound::<exceptions::ShapeError>())
        .unwrap();
    m.add(
        "StringCacheMismatchError",
        py.get_type_bound::<exceptions::StringCacheMismatchError>(),
    )
    .unwrap();
    m.add(
        "StructFieldNotFoundError",
        py.get_type_bound::<exceptions::StructFieldNotFoundError>(),
    )
    .unwrap();

    // Exceptions - Warnings
    m.add(
        "PolarsWarning",
        py.get_type_bound::<exceptions::PolarsWarning>(),
    )
    .unwrap();
    m.add(
        "PerformanceWarning",
        py.get_type_bound::<exceptions::PerformanceWarning>(),
    )
    .unwrap();
    m.add(
        "CategoricalRemappingWarning",
        py.get_type_bound::<exceptions::CategoricalRemappingWarning>(),
    )
    .unwrap();
    m.add(
        "MapWithoutReturnDtypeWarning",
        py.get_type_bound::<exceptions::MapWithoutReturnDtypeWarning>(),
    )
    .unwrap();

    // Exceptions - Panic
    m.add(
        "PanicException",
        py.get_type_bound::<pyo3::panic::PanicException>(),
    )
    .unwrap();

    // Cloud
    #[cfg(feature = "polars_cloud")]
    m.add_wrapped(wrap_pyfunction!(cloud::prepare_cloud_plan))
        .unwrap();

    // Build info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    #[cfg(feature = "build_info")]
    add_build_info(py, m)?;

    // Plugins
    #[cfg(feature = "ffi_plugin")]
    m.add_wrapped(wrap_pyfunction!(functions::register_plugin_function))
        .unwrap();

    // Capsules
    m.add("_allocator", create_allocator_capsule(py)?)?;

    Ok(())
}

#[cfg(feature = "build_info")]
fn add_build_info(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::types::{PyDict, PyString};
    let info = PyDict::new_bound(py);

    let build = PyDict::new_bound(py);
    build.set_item("rustc", build::RUSTC)?;
    build.set_item("rustc-version", build::RUSTC_VERSION)?;
    build.set_item("opt-level", build::OPT_LEVEL)?;
    build.set_item("debug", build::DEBUG)?;
    build.set_item("jobs", build::NUM_JOBS)?;
    info.set_item("compiler", build)?;

    info.set_item("time", build::BUILT_TIME_UTC)?;

    let deps = PyDict::new_bound(py);
    for (name, version) in build::DEPENDENCIES.iter() {
        deps.set_item(name, version)?;
    }
    info.set_item("dependencies", deps)?;

    let features = build::FEATURES
        .iter()
        .map(|feat| PyString::new_bound(py, feat))
        .collect::<Vec<_>>();
    info.set_item("features", features)?;

    let host = PyDict::new_bound(py);
    host.set_item("triple", build::HOST)?;
    info.set_item("host", host)?;

    let target = PyDict::new_bound(py);
    target.set_item("arch", build::CFG_TARGET_ARCH)?;
    target.set_item("os", build::CFG_OS)?;
    target.set_item("family", build::CFG_FAMILY)?;
    target.set_item("env", build::CFG_ENV)?;
    target.set_item("triple", build::TARGET)?;
    target.set_item("endianness", build::CFG_ENDIAN)?;
    target.set_item("pointer-width", build::CFG_POINTER_WIDTH)?;
    target.set_item("profile", build::PROFILE)?;
    info.set_item("target", target)?;

    let git = PyDict::new_bound(py);
    git.set_item("version", build::GIT_VERSION)?;
    git.set_item("dirty", build::GIT_DIRTY)?;
    git.set_item("hash", build::GIT_COMMIT_HASH)?;
    git.set_item("head", build::GIT_HEAD_REF)?;
    info.set_item("git", git)?;
    m.add("__build__", info)?;
    Ok(())
}
