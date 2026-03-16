use arrow::array::{MutableBinaryViewArray, Utf8ViewArray};
use polars::prelude::{ArrowDataType, IntoColumn, PlRefPath, ScanSourceRef};
use polars::series::Series;
use polars_buffer::Buffer;
use polars_core::frame::DataFrame;
use polars_error::{PolarsError, PolarsResult};
use polars_utils::python_function::PythonObject;
use pyo3::types::{PyAnyMethods, PyModule};
use pyo3::{PyErr, Python, intern};

use crate::dataframe::PyDataFrame;

pub fn call(callback: &PythonObject, paths: Buffer<PlRefPath>) -> PolarsResult<Option<DataFrame>> {
    let df = {
        let mut builder = MutableBinaryViewArray::with_capacity(
            paths.len().wrapping_mul(
                paths
                    .first()
                    .map_or(0, |x| ScanSourceRef::Path(x).to_include_path_name().len()),
            ),
        );

        for path in paths.iter() {
            builder.push_value_ignore_validity(ScanSourceRef::Path(path).to_include_path_name());
        }

        let array: Utf8ViewArray = builder.freeze_with_dtype(ArrowDataType::Utf8View);
        let c = Series::from_arrow("path".into(), Box::new(array))
            .unwrap()
            .into_column();

        DataFrame::new(paths.len(), vec![c]).unwrap()
    };

    Python::attach(|py| {
        // Wrap to Python
        let pl = PyModule::import(py, "polars")?;
        let py_df_wrapped = pl
            .getattr(intern!(py, "DataFrame"))?
            .getattr(intern!(py, "_from_pydf"))?
            .call1((PyDataFrame::new(df),))?;

        let result_wrapped = callback
            .getattr(py, intern!(py, "__call__"))?
            .call1(py, (py_df_wrapped,))?;

        if result_wrapped.is_none(py) {
            return Ok(None);
        }

        // Unwrap to Rust
        let py_pydf = result_wrapped.getattr(py, "_df").map_err(|_| {
            let pytype = result_wrapped.bind(py).get_type();
            PolarsError::ComputeError(
                format!("expected the deletion vector callback to return a 'DataFrame', got a '{pytype}'",)
                    .into(),
            )
        })?;

        let pydf = py_pydf.extract::<PyDataFrame>(py).map_err(PyErr::from)?;
        Ok(Some(pydf.df.into_inner()))
    })
}
