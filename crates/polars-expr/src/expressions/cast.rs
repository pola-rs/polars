use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;

use super::*;
use crate::expressions::{AggState, AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct CastExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) data_type: DataType,
    pub(crate) expr: Expr,
    pub(crate) options: CastOptions,
}

impl CastExpr {
    fn finish(&self, input: &Series) -> PolarsResult<Series> {
        input.cast_with_options(&self.data_type, self.options)
    }
}

impl PhysicalExpr for CastExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let series = self.input.evaluate(df, state)?;
        self.finish(&series)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;

        match ac.agg_state() {
            // this will not explode and potentially increase memory due to overlapping groups
            AggState::AggregatedList(s) => {
                let ca = s.list().unwrap();
                let casted = ca.apply_to_inner(&|s| self.finish(&s))?;
                ac.with_series(casted.into_series(), true, None)?;
            },
            AggState::AggregatedScalar(s) => {
                let s = self.finish(s)?;
                if ac.is_literal() {
                    ac.with_literal(s);
                } else {
                    ac.with_series(s, true, None)?;
                }
            },
            _ => {
                // before we flatten, make sure that groups are updated
                ac.groups();

                let s = ac.flat_naive();
                let s = self.finish(s.as_ref())?;

                if ac.is_literal() {
                    ac.with_literal(s);
                } else {
                    ac.with_series(s, false, None)?;
                }
            },
        }

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema).map(|mut fld| {
            fld.coerce(self.data_type.clone());
            fld
        })
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for CastExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let e = self.input.as_partitioned_aggregator().unwrap();
        self.finish(&e.evaluate_partitioned(df, groups, state)?)
    }

    fn finalize(
        &self,
        partitioned: Series,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let agg = self.input.as_partitioned_aggregator().unwrap();
        agg.finalize(partitioned, groups, state)
    }
}
