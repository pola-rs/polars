use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct ShiftExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) periods: i64,
    pub(crate) expr: Expr,
}

impl PhysicalExpr for ShiftExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        Ok(series.shift(self.periods))
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        // The Series are aggregate per group, then the shift is applied.
        // Because an aggregation on a next level, e.g. sum will use the group tuples to aggregate
        // and sum we must explode the Series and update the group tuples to match the new Series.
        // Because we aggregate with the current group tuples, the Series is ordered by group.
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        let s = ac
            .aggregated()
            .list()
            .unwrap()
            .apply_amortized(|s| s.as_ref().shift(self.periods).into_series())
            .into_series();
        ac.with_series(s);
        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
impl PhysicalAggregation for ShiftExpr {
    // As a final aggregation a Shift returns a list array.
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        let s = ac
            .aggregated()
            .list()
            .unwrap()
            .apply(|s| s.shift(self.periods).into_series())
            .into_series();
        Ok(Some(s))
    }
}
