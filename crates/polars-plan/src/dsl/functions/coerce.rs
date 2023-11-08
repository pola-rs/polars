#[cfg(feature = "dtype-struct")]
use super::*;

/// Take several expressions and collect them into a [`StructChunked`].
#[cfg(feature = "dtype-struct")]
pub fn as_struct(exprs: Vec<Expr>) -> Expr {
    Expr::Function {
        input: exprs,
        function: FunctionExpr::AsStruct,
        options: FunctionOptions {
            input_wildcard_expansion: true,
            pass_name_to_apply: true,
            collect_groups: ApplyOptions::ElementWise,
            ..Default::default()
        },
    }
}
