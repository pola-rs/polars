use polars::prelude::_set_check_length;
use polars_core::config::get_engine_affinity as _get_engine_affinity;
use pyo3::prelude::*;

#[pyfunction]
pub fn check_length(check: bool) {
    unsafe { _set_check_length(check) }
}

#[pyfunction]
pub fn get_engine_affinity() -> PyResult<Option<String>> {
    Ok(Some(_get_engine_affinity()))
}
