use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::error::PyPolarsErr;
use crate::PyLazyFrame;

#[pyfunction]
pub fn prepare_cloud_plan(lf: PyLazyFrame, py: Python) -> PyResult<PyObject> {
    let plan = lf.ldf.logical_plan;
    let bytes = polars::prelude::prepare_cloud_plan(plan).map_err(PyPolarsErr::from)?;

    Ok(PyBytes::new_bound(py, &bytes).to_object(py))
}
