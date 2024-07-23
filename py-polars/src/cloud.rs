use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::PyLazyFrame;

#[pyfunction]
pub fn prepare_cloud_plan(lf: PyLazyFrame, uri: String, py: Python) -> PyResult<PyObject> {
    let plan = lf.ldf.logical_plan;
    let bytes = polars::prelude::prepare_cloud_plan(plan, uri)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyBytes::new_bound(py, &bytes).to_object(py))
}
