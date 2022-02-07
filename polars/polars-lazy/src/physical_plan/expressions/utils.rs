use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;

#[allow(clippy::ptr_arg)]
pub(crate) fn as_aggregated(
    expr: &dyn PhysicalExpr,
    df: &DataFrame,
    groups: &GroupsProxy,
    state: &ExecutionState,
) -> Result<Option<Series>> {
    match expr.as_agg_expr() {
        Ok(agg_expr) => agg_expr.aggregate(df, groups, state),
        // if we have a function that is not a final aggregation, we can always evaluate the
        // function in groupby context and aggregate the result to a list
        Err(_) => {
            let mut ac = expr.evaluate_on_groups(df, groups, state)?;
            let s = ac.aggregated();
            Ok(Some(s))
        }
    }
}
