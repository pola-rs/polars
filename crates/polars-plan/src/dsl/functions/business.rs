use super::*;

#[cfg(feature = "dtype-date")]
pub fn business_day_count(start: Expr, end: Expr) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Business(BusinessFunction::BusinessDayCount {}),
        options: FunctionOptions {
            allow_rename: true,
            ..Default::default()
        },
    }
}
