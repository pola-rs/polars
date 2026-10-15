mod planner;
mod types;

use polars_error::PolarsResult;
use polars_plan::dsl::Expr;
pub(crate) use types::PlannedFunction;

/// Inspect an untyped function and determine which expressions Polars must
/// evaluate before typed JIT translation can begin.
pub(crate) fn plan(function: Expr) -> PolarsResult<PlannedFunction> {
    // Function inspection: extract initial state and referenced columns.
    planner::inspect(function)
}
