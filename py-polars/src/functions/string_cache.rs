use polars_core;
use pyo3::prelude::*;

#[pyfunction]
pub fn set_string_cache(toggle: bool) {
    polars_core::enable_string_cache(toggle)
}

#[pyfunction]
pub fn disable_string_cache() {
    polars_core::reset_string_cache()
}

#[pyfunction]
pub fn using_string_cache() -> bool {
    polars_core::using_string_cache()
}
