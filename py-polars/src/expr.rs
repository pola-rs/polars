mod binary;
mod categorical;
mod datetime;
mod general;
mod list;
#[cfg(feature = "meta")]
mod meta;
mod string;
mod r#struct;

use polars::lazy::dsl;
use pyo3::prelude::*;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: dsl::Expr,
}

impl From<dsl::Expr> for PyExpr {
    fn from(expr: dsl::Expr) -> Self {
        PyExpr { inner: expr }
    }
}
