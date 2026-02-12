#[cfg(feature = "allocator")]
pub mod allocator;

// Since Python Polars cannot share its version into here and we need to be able to build this
// package correctly without `py-polars`, we need to mirror the version here.
// example: 1.35.0-beta.1
pub static PYPOLARS_VERSION: &str = "1.38.1";

// We allow multiple features to be set simultaneously so checking with all-features
// is possible. In the case multiple are set or none at all, we set the repr to "unknown".
#[cfg(all(feature = "rtcompat", not(any(feature = "rt32", feature = "rt64"))))]
pub static RUNTIME_REPR: &str = "rtcompat";
#[cfg(all(feature = "rt32", not(any(feature = "rt64", feature = "rtcompat"))))]
pub static RUNTIME_REPR: &str = "rt32";
#[cfg(all(feature = "rt64", not(any(feature = "rt32", feature = "rtcompat"))))]
pub static RUNTIME_REPR: &str = "rt64";
#[cfg(not(any(
    all(feature = "rtcompat", not(any(feature = "rt32", feature = "rt64"))),
    all(feature = "rt32", not(any(feature = "rt64", feature = "rtcompat"))),
    all(feature = "rt64", not(any(feature = "rt32", feature = "rtcompat")))
)))]
pub static RUNTIME_REPR: &str = "unknown";

use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule};

#[cfg(feature = "catalog")]
use crate::catalog::unity::PyCatalogClient;
#[cfg(feature = "polars_cloud_client")]
use crate::cloud_client;
#[cfg(feature = "polars_cloud_server")]
use crate::cloud_server;
use crate::conversion::PyCategories;
use crate::dataframe::PyDataFrame;
use crate::expr::PyExpr;
use crate::expr::datatype::PyDataTypeExpr;
use crate::expr::selector::PySelector;
use crate::functions::PyStringCacheHolder;
#[cfg(not(target_arch = "wasm32"))]
use crate::lazyframe::PyInProcessQuery;
use crate::lazyframe::{PyLazyFrame, PyOptFlags};
use crate::lazygroupby::PyLazyGroupBy;
use crate::series::PySeries;
#[cfg(feature = "sql")]
use crate::sql::PySQLContext;
use crate::{datatypes, exceptions, extension, functions, testing};

#[pymodule(gil_used = false)] // gil_used = false will be default in PyO3 0.28.
fn _ir_nodes(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use crate::lazyframe::visitor::nodes::*;
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
    m.add_class::<MergeSorted>().unwrap();
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

#[pymodule(gil_used = false)] // gil_used = false will be default in PyO3 0.28.
fn _expr_nodes(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use crate::lazyframe::visit::PyExprIR;
    use crate::lazyframe::visitor::expr_nodes::*;
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
    m.add_class::<PyStructFunction>().unwrap();
    // Options
    m.add_class::<PyWindowMapping>().unwrap();
    m.add_class::<PyRollingGroupOptions>().unwrap();
    m.add_class::<PyGroupbyOptions>().unwrap();
    Ok(())
}

#[pymodule(gil_used = false)] // gil_used = false will be default in PyO3 0.28.
pub fn _polars_runtime(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_class::<PyLazyFrame>().unwrap();
    m.add_class::<PyOptFlags>().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    m.add_class::<PyInProcessQuery>().unwrap();
    m.add_class::<PyLazyGroupBy>().unwrap();
    m.add_class::<PyExpr>().unwrap();
    m.add_class::<PyDataTypeExpr>().unwrap();
    m.add_class::<PySelector>().unwrap();
    m.add_class::<PyStringCacheHolder>().unwrap();
    #[cfg(feature = "sql")]
    m.add_class::<PySQLContext>().unwrap();
    m.add_class::<PyCategories>().unwrap();

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
    m.add_wrapped(wrap_pyfunction!(functions::linear_space))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::linear_spaces))
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
    m.add_wrapped(wrap_pyfunction!(functions::collect_all_lazy))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::element)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::explain_all))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::collect_all_with_callback))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_lf))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::concat_arr))
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
    m.add_wrapped(wrap_pyfunction!(functions::duration))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::fold)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::lit)).unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::map_expr))
        .unwrap();
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

    // Functions: schema
    m.add_wrapped(wrap_pyfunction!(
        crate::interop::arrow::init_polars_schema_from_arrow_c_schema
    ))
    .unwrap();
    m.add_wrapped(wrap_pyfunction!(
        crate::interop::arrow::polars_schema_field_from_arrow_c_schema
    ))
    .unwrap();
    m.add_wrapped(wrap_pyfunction!(
        crate::interop::arrow::to_py::polars_schema_to_pycapsule
    ))
    .unwrap();

    // Functions: other
    m.add_wrapped(wrap_pyfunction!(functions::check_length))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::py_get_engine_affinity))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::config_reload_env_vars))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(functions::config_reload_env_var))
        .unwrap();

    #[cfg(feature = "sql")]
    m.add_wrapped(wrap_pyfunction!(functions::sql_expr))
        .unwrap();

    // Functions - I/O
    #[cfg(feature = "ipc")]
    m.add_wrapped(wrap_pyfunction!(functions::read_ipc_schema))
        .unwrap();
    #[cfg(feature = "parquet")]
    m.add_wrapped(wrap_pyfunction!(functions::read_parquet_metadata))
        .unwrap();
    #[cfg(feature = "clipboard")]
    m.add_wrapped(wrap_pyfunction!(functions::read_clipboard_string))
        .unwrap();
    #[cfg(feature = "clipboard")]
    m.add_wrapped(wrap_pyfunction!(functions::write_clipboard_string))
        .unwrap();
    #[cfg(feature = "catalog")]
    m.add_class::<PyCatalogClient>().unwrap();

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

    // Functions - escape_regex
    m.add_wrapped(wrap_pyfunction!(functions::escape_regex))
        .unwrap();

    // Dtype helpers
    m.add_wrapped(wrap_pyfunction!(datatypes::_get_dtype_max))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(datatypes::_get_dtype_min))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(datatypes::_known_timezones))
        .unwrap();

    // Extension type registry.
    m.add_wrapped(wrap_pyfunction!(extension::_register_extension_type))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(extension::_unregister_extension_type))
        .unwrap();

    // Testing
    m.add_wrapped(wrap_pyfunction!(testing::assert_series_equal_py))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(testing::assert_dataframe_equal_py))
        .unwrap();

    // Exceptions - Errors
    m.add("PolarsError", py.get_type::<exceptions::PolarsError>())
        .unwrap();
    m.add(
        "ColumnNotFoundError",
        py.get_type::<exceptions::ColumnNotFoundError>(),
    )
    .unwrap();
    m.add("ComputeError", py.get_type::<exceptions::ComputeError>())
        .unwrap();
    m.add(
        "DuplicateError",
        py.get_type::<exceptions::DuplicateError>(),
    )
    .unwrap();
    m.add(
        "InvalidOperationError",
        py.get_type::<exceptions::InvalidOperationError>(),
    )
    .unwrap();
    m.add("NoDataError", py.get_type::<exceptions::NoDataError>())
        .unwrap();
    m.add(
        "OutOfBoundsError",
        py.get_type::<exceptions::OutOfBoundsError>(),
    )
    .unwrap();
    m.add(
        "SQLInterfaceError",
        py.get_type::<exceptions::SQLInterfaceError>(),
    )
    .unwrap();
    m.add(
        "SQLSyntaxError",
        py.get_type::<exceptions::SQLSyntaxError>(),
    )
    .unwrap();
    m.add("SchemaError", py.get_type::<exceptions::SchemaError>())
        .unwrap();
    m.add(
        "SchemaFieldNotFoundError",
        py.get_type::<exceptions::SchemaFieldNotFoundError>(),
    )
    .unwrap();
    m.add("ShapeError", py.get_type::<exceptions::ShapeError>())
        .unwrap();
    m.add(
        "StringCacheMismatchError",
        py.get_type::<exceptions::StringCacheMismatchError>(),
    )
    .unwrap();
    m.add(
        "StructFieldNotFoundError",
        py.get_type::<exceptions::StructFieldNotFoundError>(),
    )
    .unwrap();

    // Exceptions - Warnings
    m.add("PolarsWarning", py.get_type::<exceptions::PolarsWarning>())
        .unwrap();
    m.add(
        "PerformanceWarning",
        py.get_type::<exceptions::PerformanceWarning>(),
    )
    .unwrap();
    m.add(
        "CategoricalRemappingWarning",
        py.get_type::<exceptions::CategoricalRemappingWarning>(),
    )
    .unwrap();
    m.add(
        "MapWithoutReturnDtypeWarning",
        py.get_type::<exceptions::MapWithoutReturnDtypeWarning>(),
    )
    .unwrap();

    // Exceptions - Panic
    m.add(
        "PanicException",
        py.get_type::<pyo3::panic::PanicException>(),
    )
    .unwrap();

    // Cloud
    #[cfg(feature = "polars_cloud_client")]
    m.add_wrapped(wrap_pyfunction!(cloud_client::prepare_cloud_plan))
        .unwrap();
    #[cfg(feature = "polars_cloud_server")]
    m.add_wrapped(wrap_pyfunction!(cloud_server::_execute_ir_plan_with_gpu))
        .unwrap();

    // Build info
    m.add("__version__", PYPOLARS_VERSION)?;
    m.add("RUNTIME_REPR", RUNTIME_REPR)?;

    // Plugins
    #[cfg(feature = "ffi_plugin")]
    m.add_wrapped(wrap_pyfunction!(functions::register_plugin_function))
        .unwrap();

    // Capsules
    #[cfg(feature = "allocator")]
    {
        m.add("_allocator", allocator::create_allocator_capsule(py)?)?;
    }

    m.add("_debug", cfg!(debug_assertions))?;

    Ok(())
}
