use std::borrow::Cow;

use polars_core::prelude::*;
use polars_plan::constants::LEN;

use super::*;
use crate::expressions::{AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct CountExpr {
    expr: Expr,
}

impl CountExpr {
    pub(crate) fn new() -> Self {
        Self { expr: Expr::Len }
    }
}

impl PhysicalExpr for CountExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Column> {
        Ok(Series::new(PlSmallStr::from_static("len"), [df.height() as IdxSize]).into_column())
    }

    fn evaluate_on_groups<'a>(
        &self,
        _df: &DataFrame,
        groups: &'a GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let ca = groups.group_count().with_name(PlSmallStr::from_static(LEN));
        let c = ca.into_column();
        Ok(AggregationContext::new(c, Cow::Borrowed(groups), true))
    }

    fn isolate_column_expr(
        &self,
        _name: &str,
    ) -> Option<(
        Arc<dyn PhysicalExpr>,
        Option<SpecializedColumnPredicateExpr>,
    )> {
        None
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(Field::new(PlSmallStr::from_static(LEN), IDX_DTYPE))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

impl PartitionedAggregation for CountExpr {
    #[allow(clippy::ptr_arg)]
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        self.evaluate_on_groups(df, groups, state)
            .map(|mut ac| ac.aggregated().into_column())
    }

    /// Called to merge all the partitioned results in a final aggregate.
    #[allow(clippy::ptr_arg)]
    fn finalize(
        &self,
        partitioned: Column,
        groups: &GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<Column> {
        // SAFETY: groups are in bounds.
        let agg = unsafe { partitioned.agg_sum(groups) };
        Ok(agg.with_name(PlSmallStr::from_static(LEN)))
    }
}
