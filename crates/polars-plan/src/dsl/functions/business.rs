use super::*;

#[cfg(feature = "dtype-date")]
pub fn business_day_count(start: Expr, end: Expr, week_mask: [bool; 7], holidays: Expr) -> Expr {
    let input = vec![start, end, holidays];

    Expr::Function {
        input,
        function: FunctionExpr::Business(BusinessFunction::BusinessDayCount { week_mask }),
    }
}
