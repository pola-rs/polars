use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::PyExpr;

#[pyfunction]
pub fn arange(start: PyExpr, end: PyExpr, step: i64) -> PyExpr {
    dsl::arange(start.inner, end.inner, step).into()
}
