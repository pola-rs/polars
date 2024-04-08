mod construction;
mod export;
mod general;
mod io;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::PyPolarsErr;

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
