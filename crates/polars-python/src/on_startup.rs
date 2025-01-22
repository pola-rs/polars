use std::any::Any;

use polars::prelude::*;
use polars_core::chunked_array::object::builder::ObjectChunkedBuilder;
use polars_core::chunked_array::object::registry::AnonymousObjectBuilder;
use polars_core::chunked_array::object::{registry, set_polars_allow_extension};
use polars_core::error::PolarsError::ComputeError;
use polars_error::{set_signals_function, PolarsWarning};
use pyo3::prelude::*;
use pyo3::{intern, IntoPyObjectExt};

use crate::dataframe::PyDataFrame;
use crate::map::lazy::{call_lambda_with_series, ToSeries};
use crate::prelude::ObjectValue;
use crate::py_modules::{pl_utils, polars};
use crate::Wrap;

fn python_function_caller_series(s: Column, lambda: &PyObject) -> PolarsResult<Column> {
    Python::with_gil(|py| {
        let object = call_lambda_with_series(py, s.clone().take_materialized_series(), lambda)
            .map_err(|s| ComputeError(format!("{}", s).into()))?;
        object.to_series(py, polars(py), s.name()).map(Column::from)
    })
}

fn python_function_caller_df(df: DataFrame, lambda: &PyObject) -> PolarsResult<DataFrame> {
    Python::with_gil(|py| {
        // create a PyDataFrame struct/object for Python
        let pydf = PyDataFrame::new(df);
        // Wrap this PyDataFrame object in the python side DataFrame wrapper
        let python_df_wrapper = polars(py)
            .getattr(py, "wrap_df")
            .unwrap()
            .call1(py, (pydf,))
            .unwrap();
        // call the lambda and get a python side df wrapper
        let result_df_wrapper = lambda.call1(py, (python_df_wrapper,)).map_err(|e| {
            PolarsError::ComputeError(format!("User provided python function failed: {e}").into())
        })?;
        // unpack the wrapper in a PyDataFrame
        let py_pydf = result_df_wrapper.getattr(py, "_df").map_err(|_| {
            let pytype = result_df_wrapper.bind(py).get_type();
            PolarsError::ComputeError(
                format!("Expected 'LazyFrame.map' to return a 'DataFrame', got a '{pytype}'",)
                    .into(),
            )
        })?;

        // Downcast to Rust
        let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
        // Finally get the actual DataFrame
        let df = pydf.df;

        Ok(df)
    })
}

fn warning_function(msg: &str, warning: PolarsWarning) {
    Python::with_gil(|py| {
        let warn_fn = pl_utils(py)
            .bind(py)
            .getattr(intern!(py, "_polars_warn"))
            .unwrap();

        if let Err(e) = warn_fn.call1((msg, Wrap(warning).into_pyobject(py).unwrap())) {
            eprintln!("{e}")
        }
    });
}

/// # Safety
/// Caller must ensure that no other threads read the objects set by this registration.
pub unsafe fn register_startup_deps(check_python_signals: bool) {
    set_polars_allow_extension(true);
    if !registry::is_object_builder_registered() {
        // Stack frames can get really large in debug mode.
        #[cfg(debug_assertions)]
        {
            recursive::set_minimum_stack_size(1024 * 1024);
            recursive::set_stack_allocation_size(1024 * 1024 * 16);
        }

        // register object type builder
        let object_builder = Box::new(|name: PlSmallStr, capacity: usize| {
            Box::new(ObjectChunkedBuilder::<ObjectValue>::new(name, capacity))
                as Box<dyn AnonymousObjectBuilder>
        });

        let object_converter = Arc::new(|av: AnyValue| {
            let object = Python::with_gil(|py| ObjectValue {
                inner: Wrap(av).into_py_any(py).unwrap(),
            });
            Box::new(object) as Box<dyn Any>
        });

        let object_size = size_of::<ObjectValue>();
        let physical_dtype = ArrowDataType::FixedSizeBinary(object_size);
        registry::register_object_builder(object_builder, object_converter, physical_dtype);
        // register SERIES UDF
        python_udf::CALL_COLUMNS_UDF_PYTHON = Some(python_function_caller_series);
        // register DATAFRAME UDF
        python_udf::CALL_DF_UDF_PYTHON = Some(python_function_caller_df);
        // register warning function for `polars_warn!`
        polars_error::set_warning_function(warning_function);

        if check_python_signals {
            fn signals_function() -> PolarsResult<()> {
                Python::with_gil(|py| {
                    py.check_signals()
                        .map_err(|err| polars_err!(ComputeError: "{err}"))
                })
            }

            set_signals_function(signals_function);
        }

        Python::with_gil(|py| {
            // init AnyValue LUT
            crate::conversion::any_value::LUT
                .set(py, Default::default())
                .unwrap();
        });
    }
}
