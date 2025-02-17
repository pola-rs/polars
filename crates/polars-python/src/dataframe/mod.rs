#[cfg(feature = "pymethods")]
mod construction;
#[cfg(feature = "pymethods")]
mod export;
#[cfg(feature = "pymethods")]
mod general;
#[cfg(feature = "pymethods")]
mod io;
#[cfg(feature = "pymethods")]
mod serde;

use polars::prelude::DataFrame;
use pyo3::pyclass;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataFrame {
    pub df: DataFrame,
}

impl From<DataFrame> for PyDataFrame {
    fn from(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}

impl PyDataFrame {
    pub(crate) fn new(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}
