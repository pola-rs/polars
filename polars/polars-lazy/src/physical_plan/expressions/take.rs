use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

pub struct TakeExpr {
    pub(crate) expr: Arc<dyn PhysicalExpr>,
    pub(crate) idx: Arc<dyn PhysicalExpr>,
}

impl TakeExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, idx: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr, idx }
    }
    fn finish(&self, df: &DataFrame, state: &ExecutionState, series: Series) -> Result<Series> {
        let idx = self.idx.evaluate(df, state)?;
        let idx_ca = idx.u32()?;

        series.take(idx_ca)
    }
}

impl PhysicalExpr for TakeExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.expr.evaluate(df, state)?;
        self.finish(df, state, series)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let (series, groups) = self.expr.evaluate_on_groups(df, groups, state)?;
        Ok((self.finish(df, state, series)?, groups))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.expr.to_field(input_schema)
    }
}
