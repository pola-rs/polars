mod planner;
mod types;

pub(crate) use planner::VALUE_NAME;
use polars_error::PolarsResult;
use polars_plan::dsl::Expr;
pub(crate) use types::PlannedFunction;

/// Inspect an untyped function and determine which expressions Polars must
/// evaluate before typed JIT translation can begin. When `input` is given, the fold runs over
/// its rows and the function reads its current value through [`VALUE_NAME`].
pub(crate) fn plan(function: Expr, input: Option<Expr>) -> PolarsResult<PlannedFunction> {
    // Function inspection: extract initial state and referenced columns.
    planner::inspect(function, input)
}
