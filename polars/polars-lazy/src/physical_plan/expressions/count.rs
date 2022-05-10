use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::prelude::*;
use polars_core::utils::NoNull;
use std::borrow::Cow;

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
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        Ok(Series::new("count", [df.height() as IdxSize]))
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        groups: &'a GroupsProxy,
        _state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ca = match groups {
            GroupsProxy::Idx(groups) => {
                let ca: NoNull<IdxCa> = groups
                    .all()
                    .iter()
                    .map(|g| g.len() as IdxSize)
                    .collect_trusted();
                ca.into_inner()
            }
            GroupsProxy::Slice(groups) => {
                let ca: NoNull<IdxCa> = groups.iter().map(|g| g[1]).collect_trusted();
                ca.into_inner()
            }
        };
        ca.rename(COUNT_NAME);
        let s = ca.into_series();

        Ok(AggregationContext::new(s, Cow::Borrowed(groups), true))
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("count", DataType::UInt32))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for CountExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Series> {
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
    ) -> Result<Series> {
        let mut agg = partitioned.agg_sum(groups);
        agg.rename(COUNT_NAME);
        Ok(agg)
    }
}
