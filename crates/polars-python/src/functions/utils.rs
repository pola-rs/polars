use polars::prelude::_set_check_length;
use polars_core::config::use_gpu_engine as _use_gpu_engine;
use pyo3::prelude::*;

#[pyfunction]
pub fn check_length(check: bool) {
    unsafe { _set_check_length(check) }
}

#[pyfunction]
pub fn use_gpu_engine() -> PyResult<Option<bool>> {
    Ok(Some(_use_gpu_engine()))
}
