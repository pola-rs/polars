use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::PyLazyFrame;

#[pyfunction]
pub fn prepare_cloud_plan(lf: PyLazyFrame, uri: String) -> PyResult<()> {
    let plan = lf.ldf.logical_plan;
    polars::prelude::prepare_cloud_plan(plan, uri).map_err(|e| PyValueError::new_err(e.to_string()))
}
