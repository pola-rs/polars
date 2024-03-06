use polars::lazy::dsl;
use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::PyExpr;

#[pyfunction]
pub fn all_horizontal(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::all_horizontal(exprs).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn any_horizontal(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::any_horizontal(exprs).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn max_horizontal(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::max_horizontal(exprs).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn min_horizontal(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::min_horizontal(exprs).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn sum_horizontal(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::sum_horizontal(exprs).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn mean_horizontal(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::mean_horizontal(exprs).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}
