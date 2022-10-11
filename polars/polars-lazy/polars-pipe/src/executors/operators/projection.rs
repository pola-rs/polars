use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;

use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

pub(crate) struct FastProjectionOperator {
    pub(crate) columns: Arc<Vec<Arc<str>>>,
}

impl Operator for FastProjectionOperator {
    fn execute(
        &self,
        _context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let chunk = chunk.with_data(chunk.data.select(self.columns.as_slice())?);
        Ok(OperatorResult::Finished(chunk))
    }
}

pub(crate) struct ProjectionOperator {
    pub(crate) exprs: Vec<Arc<dyn PhysicalPipedExpr>>,
}

impl Operator for ProjectionOperator {
    fn execute(
        &self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let projected = self
            .exprs
            .iter()
            .map(|e| e.evaluate(chunk, context.execution_state.as_ref()))
            .collect::<PolarsResult<Vec<_>>>()?;

        let chunk = chunk.with_data(DataFrame::new_no_checks(projected));
        Ok(OperatorResult::Finished(chunk))
    }
}
