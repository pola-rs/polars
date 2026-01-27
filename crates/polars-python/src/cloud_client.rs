use pyo3::prelude::{Python, *};
use pyo3::types::PyBytes;

use crate::PyLazyFrame;
use crate::error::PyPolarsErr;

#[pyfunction]
pub fn prepare_cloud_plan(
    lf: PyLazyFrame,
    allow_local_scans: bool,
    py: Python<'_>,
) -> PyResult<(Bound<'_, PyBytes>, u32)> {
    let lf = lf.ldf.into_inner();
    let opt_flags = lf.get_current_optimizations();
    let plan = lf.logical_plan;
    let bytes =
        polars::prelude::prepare_cloud_plan(plan, allow_local_scans).map_err(PyPolarsErr::from)?;

    Ok((PyBytes::new(py, &bytes), opt_flags.bits()))
}
