#[cfg(any(feature = "range", feature = "arg_where"))]
use super::*;

/// Find the indexes that would sort these series in order of appearance.
/// That means that the first `Series` will be used to determine the ordering
/// until duplicates are found. Once duplicates are found, the next `Series` will
/// be used and so on.
#[cfg(feature = "range")]
pub fn arg_sort_by<E: AsRef<[Expr]>>(by: E, descending: &[bool]) -> Expr {
    let e = &by.as_ref()[0];
    let name = expr_output_name(e).unwrap();
    int_range(lit(0 as IdxSize), count().cast(IDX_DTYPE), 1)
        .sort_by(by, descending)
        .alias(name.as_ref())
}

#[cfg(feature = "arg_where")]
/// Get the indices where `condition` evaluates `true`.
pub fn arg_where<E: Into<Expr>>(condition: E) -> Expr {
    let condition = condition.into();
    Expr::Function {
        input: vec![condition],
        function: FunctionExpr::ArgWhere,
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            ..Default::default()
        },
    }
}
