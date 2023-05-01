mod binary;
mod categorical;
mod datetime;
mod general;
mod list;
#[cfg(feature = "meta")]
mod meta;
mod string;
mod r#struct;

use polars::lazy::dsl::Expr;
use pyo3::prelude::*;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: Expr,
}

impl From<Expr> for PyExpr {
    fn from(expr: Expr) -> Self {
        PyExpr { inner: expr }
    }
}
