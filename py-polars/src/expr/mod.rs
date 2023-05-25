mod array;
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

pub(crate) trait ToExprs {
    fn to_exprs(self) -> Vec<Expr>;
}

impl ToExprs for Vec<PyExpr> {
    fn to_exprs(self) -> Vec<Expr> {
        // Safety
        // repr is transparent
        unsafe { std::mem::transmute(self) }
    }
}

pub(crate) trait ToPyExprs {
    fn to_pyexprs(self) -> Vec<PyExpr>;
}

impl ToPyExprs for Vec<Expr> {
    fn to_pyexprs(self) -> Vec<PyExpr> {
        // Safety
        // repr is transparent
        unsafe { std::mem::transmute(self) }
    }
}
