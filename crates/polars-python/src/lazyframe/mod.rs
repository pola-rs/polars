mod exitable;
#[cfg(feature = "pymethods")]
mod general;
#[cfg(feature = "pymethods")]
mod serde;
pub mod visit;
pub mod visitor;

pub use exitable::PyInProcessQuery;
use polars::prelude::LazyFrame;
use pyo3::pyclass;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyLazyFrame {
    pub ldf: LazyFrame,
}

impl From<LazyFrame> for PyLazyFrame {
    fn from(ldf: LazyFrame) -> Self {
        PyLazyFrame { ldf }
    }
}
