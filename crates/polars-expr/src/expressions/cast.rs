use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;

use super::*;
use crate::expressions::{AggState, AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct CastExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) dtype: DataType,
    pub(crate) expr: Expr,
    pub(crate) options: CastOptions,
}

impl CastExpr {
    fn finish(&self, input: &Column) -> PolarsResult<Column> {
        input.cast_with_options(&self.dtype, self.options)
    }
}

impl PhysicalExpr for CastExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let column = self.input.evaluate(df, state)?;
        self.finish(&column)
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
                let casted = ca.apply_to_inner(&|s| {
                    self.finish(&s.into_column())
                        .map(|c| c.take_materialized_series())
                })?;
                ac.with_series(casted.into_series(), true, None)?;
            },
            AggState::AggregatedScalar(s) => {
                let s = self.finish(&s.clone().into_column())?;
                if ac.is_literal() {
                    ac.with_literal(s.take_materialized_series());
                } else {
                    ac.with_series(s.take_materialized_series(), true, None)?;
                }
            },
            _ => {
                // before we flatten, make sure that groups are updated
                ac.groups();

                let s = ac.flat_naive();
                let s = self.finish(&s.as_ref().clone().into_column())?;

                if ac.is_literal() {
                    ac.with_literal(s.take_materialized_series());
                } else {
                    ac.with_series(s.take_materialized_series(), false, None)?;
                }
            },
        }

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema).map(|mut fld| {
            fld.coerce(self.dtype.clone());
            fld
        })
    }

    fn is_scalar(&self) -> bool {
        self.input.is_scalar()
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
    ) -> PolarsResult<Column> {
        let e = self.input.as_partitioned_aggregator().unwrap();
        self.finish(&e.evaluate_partitioned(df, groups, state)?)
    }

    fn finalize(
        &self,
        partitioned: Column,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let agg = self.input.as_partitioned_aggregator().unwrap();
        agg.finalize(partitioned, groups, state)
    }
}
