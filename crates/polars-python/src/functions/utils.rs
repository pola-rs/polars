use polars::prelude::_set_check_length;
use polars_core::config::get_engine_affinity;
use pyo3::prelude::*;

use crate::lazyframe::visit::get_ir_version;

#[pyfunction]
pub fn check_length(check: bool) {
    unsafe { _set_check_length(check) }
}

#[pyfunction(name = "get_engine_affinity")]
pub fn py_get_engine_affinity() -> PyResult<String> {
    Ok(get_engine_affinity())
}

#[pyfunction(name = "get_ir_version")]
pub fn py_get_ir_version() -> PyResult<(u16, u16)> {
    Ok(get_ir_version())
}
