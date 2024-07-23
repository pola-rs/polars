use super::*;

/// Take several expressions and collect them into a [`StructChunked`].
/// # Panics
/// panics if `exprs` is empty.
pub fn as_struct(exprs: Vec<Expr>) -> Expr {
    assert!(
        !exprs.is_empty(),
        "expected at least 1 field in 'as_struct'"
    );
    Expr::Function {
        input: exprs,
        function: FunctionExpr::AsStruct,
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default()
                | FunctionFlags::PASS_NAME_TO_APPLY
                | FunctionFlags::INPUT_WILDCARD_EXPANSION,
            ..Default::default()
        },
    }
}
