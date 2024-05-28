use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::PyExpr;

#[pyfunction]
pub fn business_day_count(
    start: PyExpr,
    end: PyExpr,
    week_mask: [bool; 7],
    holidays: Vec<i32>,
) -> PyExpr {
    let start = start.inner;
    let end = end.inner;
    dsl::business_day_count(start, end, week_mask, holidays).into()
}
