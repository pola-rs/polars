use polars_core::prelude::*;

use super::*;
use crate::expressions::{AggregationContext, PartitionedAggregation, PhysicalExpr};

pub struct AliasExpr {
    pub(crate) physical_expr: Arc<dyn PhysicalExpr>,
    pub(crate) name: PlSmallStr,
    expr: Expr,
}

impl AliasExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, name: PlSmallStr, expr: Expr) -> Self {
        Self {
            physical_expr,
            name,
            expr,
        }
    }

    fn finish(&self, input: Column) -> Column {
        input.with_name(self.name.clone())
    }
}

impl PhysicalExpr for AliasExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let series = self.physical_expr.evaluate(df, state)?;
        Ok(self.finish(series))
    }

    fn evaluate_inline_impl(&self, depth_limit: u8) -> Option<Column> {
        let depth_limit = depth_limit.checked_sub(1)?;
        self.physical_expr
            .evaluate_inline_impl(depth_limit)
            .map(|s| self.finish(s))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let c = ac.take();
        let c = self.finish(c);

        if ac.is_literal() {
            ac.with_literal(c);
        } else {
            ac.with_values(c, ac.is_aggregated(), Some(&self.expr))?;
        }
        Ok(ac)
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

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        Ok(Field::new(
            self.name.clone(),
            self.physical_expr.to_field(input_schema)?.dtype().clone(),
        ))
    }

    fn is_literal(&self) -> bool {
        self.physical_expr.is_literal()
    }

    fn is_scalar(&self) -> bool {
        self.physical_expr.is_scalar()
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for AliasExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let agg = self.physical_expr.as_partitioned_aggregator().unwrap();
        let s = agg.evaluate_partitioned(df, groups, state)?;
        Ok(s.with_name(self.name.clone()))
    }

    fn finalize(
        &self,
        partitioned: Column,
        groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let agg = self.physical_expr.as_partitioned_aggregator().unwrap();
        let s = agg.finalize(partitioned, groups, state)?;
        Ok(s.with_name(self.name.clone()))
    }
}
