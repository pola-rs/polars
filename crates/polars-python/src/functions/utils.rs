use polars::prelude::_set_check_length;
use polars_core::config::get_default_engine as _get_default_engine;
use pyo3::prelude::*;

#[pyfunction]
pub fn check_length(check: bool) {
    unsafe { _set_check_length(check) }
}

#[pyfunction]
pub fn get_default_engine() -> PyResult<Option<String>> {
    Ok(Some(_get_default_engine()))
}
