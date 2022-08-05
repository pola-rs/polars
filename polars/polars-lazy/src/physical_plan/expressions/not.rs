use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct NotExpr(Arc<dyn PhysicalExpr>, Expr);

impl NotExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self(physical_expr, expr)
    }

    fn finish(&self, series: Series) -> Result<Series> {
        if let Ok(ca) = series.bool() {
            Ok((!ca).into_series())
        } else {
            Err(PolarsError::InvalidOperation(
                format!("NotExpr expected a boolean type, got: {:?}", series).into(),
            ))
        }
    }
}
impl PhysicalExpr for NotExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.1)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.0.evaluate(df, state)?;
        self.finish(series)
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut ac = self.0.evaluate_on_groups(df, groups, state)?;
        let s = ac.flat_naive().into_owned();
        ac.with_series(self.finish(s)?, false);

        Ok(ac)
    }

    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("not", DataType::Boolean))
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
    fn is_valid_aggregation(&self) -> bool {
        self.0.is_valid_aggregation()
    }
}

impl PartitionedAggregation for NotExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Series> {
        let a = self.0.as_partitioned_aggregator().unwrap();
        let s = a.evaluate_partitioned(df, groups, state)?;
        self.finish(s)
    }

    fn finalize(
        &self,
        partitioned: Series,
        _groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> Result<Series> {
        Ok(partitioned)
    }
}
