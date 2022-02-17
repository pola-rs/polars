use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use crate::utils::expr_to_root_column_name;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_io::predicates::StatsEvaluator;
#[cfg(feature = "parquet")]
use polars_io::prelude::predicates::BatchStats;
use std::sync::Arc;

pub struct IsNullExpr {
    physical_expr: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl IsNullExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self {
            physical_expr,
            expr,
        }
    }
}

impl PhysicalExpr for IsNullExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.physical_expr.evaluate(df, state)?;
        Ok(series.is_null().into_series())
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.physical_expr.evaluate_on_groups(df, groups, state)?;
        let s = ac.flat_naive();
        let s = s.is_null().into_series();
        ac.with_series(s, false);

        Ok(ac)
    }

    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("is_null", DataType::Boolean))
    }
    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        Some(self)
    }
}

#[cfg(feature = "parquet")]
impl StatsEvaluator for IsNullExpr {
    fn should_read(&self, stats: &BatchStats) -> Result<bool> {
        let root = expr_to_root_column_name(&self.expr)?;

        let read = true;
        let skip = false;

        match stats.get_stats(&root).ok() {
            Some(st) => match st.null_count() {
                Some(0) => Ok(skip),
                _ => Ok(read),
            },
            None => Ok(read),
        }
    }
}
