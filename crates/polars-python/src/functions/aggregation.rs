use polars::lazy::dsl;
use polars::prelude::QuantileFractionDtypes;
use pyo3::prelude::*;

use crate::error::PyPolarsErr;
use crate::expr::ToExprs;
use crate::{PyExpr, PySeries};

/// Checks a constant `quantile` fraction, which mustn't be null, and returns its values.
/// `fraction` holds one row, and `allow_list` says whether that row may be a list of fractions.
#[pyfunction]
pub fn extract_quantiles(fraction: PySeries, allow_list: bool) -> PyResult<Vec<f64>> {
    let allowed = match allow_list {
        true => QuantileFractionDtypes::NumberOrList,
        false => QuantileFractionDtypes::Number,
    };
    let fraction = fraction.series.into_inner();
    let quantiles = polars::prelude::extract_non_null_quantiles(&fraction, allowed)
        .map_err(PyPolarsErr::from)?;
    Ok(quantiles)
}

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
pub fn sum_horizontal(exprs: Vec<PyExpr>, ignore_nulls: bool) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::sum_horizontal(exprs, ignore_nulls).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn cum_sum_horizontal(exprs: Vec<PyExpr>) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::cum_sum_horizontal(exprs).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}

#[pyfunction]
pub fn mean_horizontal(exprs: Vec<PyExpr>, ignore_nulls: bool) -> PyResult<PyExpr> {
    let exprs = exprs.to_exprs();
    let e = dsl::mean_horizontal(exprs, ignore_nulls).map_err(PyPolarsErr::from)?;
    Ok(e.into())
}
