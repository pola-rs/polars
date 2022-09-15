use std::borrow::Cow;

use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

const COUNT_NAME: &str = "count";

pub struct CountExpr {
    expr: Expr,
}

impl CountExpr {
    pub(crate) fn new() -> Self {
        Self { expr: Expr::Count }
    }
}

impl PhysicalExpr for CountExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }
    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Series> {
        Ok(Series::new("count", [df.height() as IdxSize]))
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        groups: &'a GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ca = groups.group_count();
        ca.rename(COUNT_NAME);
        let s = ca.into_series();

        Ok(AggregationContext::new(s, Cow::Borrowed(groups), true))
    }
    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(Field::new("count", DataType::UInt32))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn is_valid_aggregation(&self) -> bool {
        true
    }
}

impl PartitionedAggregation for CountExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        self.evaluate_on_groups(df, groups, state)
            .map(|mut ac| ac.aggregated())
    }

    /// Called to merge all the partitioned results in a final aggregate.
    #[allow(clippy::ptr_arg)]
    fn finalize(
        &self,
        partitioned: Series,
        groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        // safety:
        // groups are in bounds
        let mut agg = unsafe { partitioned.agg_sum(groups) };
        agg.rename(COUNT_NAME);
        Ok(agg)
    }
}
