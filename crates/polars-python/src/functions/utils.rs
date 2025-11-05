use polars::prelude::_set_check_length;
use polars_core::config::get_engine_affinity;
use pyo3::prelude::*;

#[pyfunction]
pub fn check_length(check: bool) {
    unsafe { _set_check_length(check) }
}

#[pyfunction(name = "get_engine_affinity")]
pub fn py_get_engine_affinity() -> PyResult<String> {
    Ok(get_engine_affinity())
}
