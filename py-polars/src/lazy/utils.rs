use polars::lazy::dsl::Expr;

use crate::lazy::dsl::PyExpr;

pub fn py_exprs_to_exprs(py_exprs: Vec<PyExpr>) -> Vec<Expr> {
    // Safety:
    // transparent struct
    unsafe { std::mem::transmute(py_exprs) }
}
