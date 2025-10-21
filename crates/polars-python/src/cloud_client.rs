use pyo3::prelude::{Python, *};
use pyo3::types::PyBytes;

use crate::PyLazyFrame;
use crate::error::PyPolarsErr;

#[pyfunction]
pub fn prepare_cloud_plan(lf: PyLazyFrame, py: Python<'_>) -> PyResult<Bound<'_, PyBytes>> {
    let plan = lf.ldf.into_inner().logical_plan;
    let bytes = polars::prelude::prepare_cloud_plan(plan).map_err(PyPolarsErr::from)?;

    Ok(PyBytes::new(py, &bytes))
}
