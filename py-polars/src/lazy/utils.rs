use crate::lazy::dsl::PyExpr;
use polars::lazy::dsl::Expr;

pub fn py_exprs_to_exprs(py_exprs: Vec<PyExpr>) -> Vec<Expr> {
    py_exprs.into_iter().map(|expr| expr.inner).collect()
}
