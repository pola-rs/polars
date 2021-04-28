use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct TakeExpr {
    pub(crate) expr: Arc<dyn PhysicalExpr>,
    pub(crate) idx: Arc<dyn PhysicalExpr>,
}

impl TakeExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, idx: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr, idx }
    }
}

impl PhysicalExpr for TakeExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.expr.evaluate(df, state)?;
        let idx = self.idx.evaluate(df, state)?;
        let idx_ca = idx.u32()?;

        Ok(series.take(idx_ca))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.expr.to_field(input_schema)
    }
}
