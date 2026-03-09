use polars_core::frame::DataFrame;
use polars_error::{PolarsError, PolarsResult};
use polars_utils::python_function::PythonObject;
use pyo3::types::PyAnyMethods;
use pyo3::{PyErr, Python, intern};

use crate::dataframe::PyDataFrame;

pub fn call(callback: &PythonObject) -> PolarsResult<Option<DataFrame>> {
    Python::attach(|py| {
        let result_wrapped = callback.getattr(py, intern!(py, "__call__"))?.call0(py)?;

        if result_wrapped.is_none(py) {
            return Ok(None);
        }

        // unpack the wrapper in a PyDataFrame
        let py_pydf = result_wrapped.getattr(py, "_df").map_err(|_| {
            let pytype = result_wrapped.bind(py).get_type();
            PolarsError::ComputeError(
                format!("Expected the call to deletion_vectors() to return a 'DataFrame', got a '{pytype}'",)
                    .into(),
            )
        })?;

        // Downcast to Rust
        let pydf = py_pydf
            .extract::<PyDataFrame>(py)
            .map_err(|e| PyErr::from(e))?;
        Ok(Some(pydf.df.into_inner()))
    })
}
