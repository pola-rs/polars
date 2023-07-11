use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::expr::ToExprs;
use crate::PyExpr;

#[pyfunction]
pub fn all_horizontal(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::all_horizontal(exprs).into()
}

#[pyfunction]
pub fn any_horizontal(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::any_horizontal(exprs).into()
}

#[pyfunction]
pub fn max_horizontal(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::max_horizontal(exprs).into()
}

#[pyfunction]
pub fn min_horizontal(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::min_horizontal(exprs).into()
}

#[pyfunction]
pub fn sum_horizontal(exprs: Vec<PyExpr>) -> PyExpr {
    let exprs = exprs.to_exprs();
    dsl::sum_horizontal(exprs).into()
}
