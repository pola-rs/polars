use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;

use crate::PyLazyFrame;

#[pyfunction]
pub fn assert_cloud_eligible(lf: PyLazyFrame) -> PyResult<()> {
    let plan = &lf.ldf.logical_plan;
    polars::prelude::assert_cloud_eligible(plan)
        .map_err(|e| PyAssertionError::new_err(e.to_string()))?;
    Ok(())
}
