use polars::prelude::_set_check_length;
use pyo3::prelude::*;

#[pyfunction]
pub fn check_length(check: bool) {
    unsafe { _set_check_length(check) }
}
