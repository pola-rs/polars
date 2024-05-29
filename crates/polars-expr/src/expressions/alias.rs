use polars_core::prelude::*;

use super::*;
use crate::expressions::{AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct AliasExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    pub(crate) name: Arc<str>,
    expr: Expr,
}

impl AliasExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, name: Arc<str>, expr: Expr) -> Self {
        Self {
            physical_expr,
            name,
            expr,
        }
    }

    fn finish(&self, input: Series) -> Series {
        input.with_name(&self.name)
    }
}

impl PhysicalExpr for AliasExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let series = self.physical_expr.evaluate(df, state)?;
        Ok(self.finish(series))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let s = ac.take();
        let s = self.finish(s);

        if ac.is_literal() {
            ac.with_literal(s);
        } else {
            ac.with_series(s, ac.is_aggregated(), Some(&self.expr))?;
        }
        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        Ok(Field::new(
            &self.name,
            self.physical_expr
                .to_field(input_schema)?
                .data_type()
                .clone(),
        ))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for AliasExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let agg = self.physical_expr.as_partitioned_aggregator().unwrap();
        let s = agg.evaluate_partitioned(df, groups, state)?;
        Ok(s.with_name(&self.name))
    }

    fn finalize(
        &self,
        partitioned: Series,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let agg = self.physical_expr.as_partitioned_aggregator().unwrap();
        let s = agg.finalize(partitioned, groups, state)?;
        Ok(s.with_name(&self.name))
    }
}
