use polars_core;
use pyo3::prelude::*;

#[pyfunction]
pub fn _set_string_cache(active: bool) {
    polars_core::_set_string_cache(active)
}

#[pyfunction]
pub fn enable_string_cache() {
    polars_core::enable_string_cache()
}

#[pyfunction]
pub fn disable_string_cache() {
    polars_core::disable_string_cache()
}

#[pyfunction]
pub fn using_string_cache() -> bool {
    polars_core::using_string_cache()
}
