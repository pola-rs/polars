use super::*;

/// Create a column of length `n` containing `n` copies of the literal `value`.
///
/// Generally you won't need this function, as `lit(value)` already represents a column containing
/// only `value` whose length is automatically set to the correct number of rows.
pub fn repeat<E: Into<Expr>>(value: E, n: Expr) -> Expr {
    let input = vec![value.into(), n];

    let expr = Expr::Function {
        input,
        function: FunctionExpr::Repeat,
        options: FunctionOptions {
            flags: FunctionFlags::default()
                | FunctionFlags::ALLOW_RENAME
                | FunctionFlags::CHANGES_LENGTH,
            ..Default::default()
        },
    };

    // @NOTE: This alias should probably not be here for consistency, but it is here for backwards
    // compatibility until 2.0.
    expr.alias(PlSmallStr::from_static("repeat"))
}
