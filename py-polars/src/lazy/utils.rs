use polars::lazy::dsl::Expr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::lazy::dsl::PyExpr;

pub fn py_exprs_to_exprs(py_exprs: Vec<PyExpr>) -> Vec<Expr> {
    // Safety:
    // transparent struct
    unsafe { std::mem::transmute(py_exprs) }
}

pub fn check_strptime_format(format: Option<&str>) -> PyResult<()> {
    if let Some(format) = format {
        if format.contains("%f") {
            let message = "directive '%f' is not supported in Python Polars, as it differs from the Python standard library.\n\
                Instead, please use one of:\n\
                 - '%.f'\n\
                 - '%3f'\n\
                 - '%6f'\n\
                 - '%9f'";
            return Err(PyValueError::new_err(message));
        }
    }
    Ok(())
}
