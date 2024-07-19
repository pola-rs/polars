use super::*;

/// Take several expressions and collect them into a [`StructChunked`].
pub fn as_struct(exprs: Vec<Expr>) -> Expr {
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
