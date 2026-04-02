use super::*;

/// Take several expressions and collect them into a [`StructChunked`].
/// # Panics
/// panics if `exprs` is empty.
pub fn as_struct(exprs: Vec<Expr>) -> Expr {
    assert!(
        !exprs.is_empty(),
        "expected at least 1 field in 'as_struct'"
    );
    Expr::n_ary(FunctionExpr::AsStruct, exprs)
}
