use super::*;

/// Create a column of length `n` containing `n` copies of the literal `value`.
///
/// Generally you won't need this function, as `lit(value)` already represents a column containing
/// only `value` whose length is automatically set to the correct number of rows.
pub fn repeat<E: Into<Expr>>(value: E, n: Expr) -> Expr {
    Expr::n_ary(FunctionExpr::Repeat, vec![value.into(), n])
}
