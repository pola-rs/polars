use polars_core::StringCacheHolder;
use pyo3::prelude::*;

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

#[pyclass]
pub struct PyStringCacheHolder {
    _inner: StringCacheHolder,
}

#[pymethods]
impl PyStringCacheHolder {
    #[new]
    fn new() -> Self {
        Self {
            _inner: StringCacheHolder::hold(),
        }
    }
}
