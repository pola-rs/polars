use std::any::Any;
use std::sync::Arc;

use polars::prelude::*;
use polars_core::chunked_array::object::builder::ObjectChunkedBuilder;
use polars_core::chunked_array::object::registry;
use polars_core::chunked_array::object::registry::AnonymousObjectBuilder;
use polars_core::error::PolarsError::ComputeError;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_error::PolarsWarning;
use pyo3::intern;
use pyo3::prelude::*;

use crate::dataframe::PyDataFrame;
use crate::map::lazy::{call_lambda_with_series, ToSeries};
use crate::prelude::{python_udf, ObjectValue};
use crate::py_modules::{POLARS, UTILS};
use crate::Wrap;

fn python_function_caller_series(s: Series, lambda: &PyObject) -> PolarsResult<Series> {
    Python::with_gil(|py| {
        let object = call_lambda_with_series(py, s.clone(), lambda)
            .map_err(|s| ComputeError(format!("{}", s).into()))?;
        object.to_series(py, &POLARS, s.name())
    })
}

fn python_function_caller_df(df: DataFrame, lambda: &PyObject) -> PolarsResult<DataFrame> {
    Python::with_gil(|py| {
        // create a PyDataFrame struct/object for Python
        let pydf = PyDataFrame::new(df);
        // Wrap this PyDataFrame object in the python side DataFrame wrapper
        let python_df_wrapper = POLARS
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
            let pytype = result_df_wrapper.as_ref(py).get_type();
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
        let warn_fn = UTILS
            .as_ref(py)
            .getattr(intern!(py, "_polars_warn"))
            .unwrap();

        if let Err(e) = warn_fn.call1((msg, Wrap(warning))) {
            eprintln!("{e}")
        }
    });
}

#[pyfunction]
pub fn __register_startup_deps() {
    if !registry::is_object_builder_registered() {
        // register object type builder
        let object_builder = Box::new(|name: &str, capacity: usize| {
            Box::new(ObjectChunkedBuilder::<ObjectValue>::new(name, capacity))
                as Box<dyn AnonymousObjectBuilder>
        });

        let object_converter = Arc::new(|av: AnyValue| {
            let object = Python::with_gil(|py| ObjectValue {
                inner: Wrap(av).to_object(py),
            });
            Box::new(object) as Box<dyn Any>
        });

        registry::register_object_builder(object_builder, object_converter);
        // register SERIES UDF
        unsafe { python_udf::CALL_SERIES_UDF_PYTHON = Some(python_function_caller_series) }
        // register DATAFRAME UDF
        unsafe { python_udf::CALL_DF_UDF_PYTHON = Some(python_function_caller_df) }
        // register warning function for `polars_warn!`
        unsafe { polars_error::set_warning_function(warning_function) };
        Python::with_gil(|py| {
            // init AnyValue LUT
            crate::conversion::LUT.set(py, Default::default()).unwrap();
        });
    }
}
