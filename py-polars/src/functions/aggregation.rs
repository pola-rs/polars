use std::sync::Arc;

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

#[cfg(feature = "ffi_plugin")]
#[pyfunction]
pub fn register_plugin(
    lib: &str,
    symbol: &str,
    args: Vec<PyExpr>,
    kwargs: Vec<u8>,
    is_elementwise: bool,
    input_wildcard_expansion: bool,
    returns_scalar: bool,
    cast_to_supertypes: bool,
    pass_name_to_apply: bool,
    changes_length: bool,
) -> PyResult<PyExpr> {
    use polars_plan::prelude::*;

    let collect_groups = if is_elementwise {
        ApplyOptions::ElementWise
    } else {
        ApplyOptions::GroupWise
    };
    let mut input = Vec::with_capacity(args.len());
    for a in args {
        input.push(a.inner)
    }

    Ok(Expr::Function {
        input,
        function: FunctionExpr::FfiPlugin {
            lib: Arc::from(lib),
            symbol: Arc::from(symbol),
            kwargs: Arc::from(kwargs),
        },
        options: FunctionOptions {
            collect_groups,
            input_wildcard_expansion,
            returns_scalar,
            cast_to_supertypes,
            pass_name_to_apply,
            changes_length,
            ..Default::default()
        },
    }
    .into())
}
