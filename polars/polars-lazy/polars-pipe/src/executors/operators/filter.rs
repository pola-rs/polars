use std::sync::Arc;

use polars_core::error::{PolarsError, PolarsResult};

use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

pub(crate) struct FilterOperator {
    pub(crate) predicate: Arc<dyn PhysicalPipedExpr>,
}

impl Operator for FilterOperator {
    fn execute(
        &self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let s = self.predicate.evaluate(chunk, &context.execution_state)?;
        let mask = s.bool().map_err(|e| {
            PolarsError::ComputeError(
                format!("Filter predicate must be of type Boolean, got: {:?}", e).into(),
            )
        })?;
        // TODO! filter sequentially?
        let df = chunk.data.filter(mask)?;

        Ok(OperatorResult::Finished(chunk.with_data(df)))
    }
}
