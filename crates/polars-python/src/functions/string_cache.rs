use pyo3::prelude::*;

#[pyfunction]
pub fn enable_string_cache() {
    // The string cache no longer exists.
}

#[pyfunction]
pub fn disable_string_cache() {
    // The string cache no longer exists.
}

#[pyfunction]
pub fn using_string_cache() -> bool {
    // The string cache no longer exists.
    true
}

#[pyclass(frozen)]
pub struct PyStringCacheHolder;

#[pymethods]
impl PyStringCacheHolder {
    #[new]
    fn new() -> Self {
        Self
    }
}
