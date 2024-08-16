mod exitable;
mod general;
mod serde;
pub mod visit;
pub mod visitor;

pub use exitable::PyInProcessQuery;
use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::{PyDataFrame, PyExpr};

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
