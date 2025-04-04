use super::*;

#[cfg(feature = "dtype-date")]
pub fn business_day_count(
    start: Expr,
    end: Expr,
    week_mask: [bool; 7],
    holidays: Vec<i32>,
) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::Business(BusinessFunction::BusinessDayCount {
            week_mask,
            holidays,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default() | FunctionFlags::ALLOW_RENAME,
            ..Default::default()
        },
    }
}
