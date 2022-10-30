use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;

use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub(crate) struct FastProjectionOperator {
    pub(crate) columns: Arc<Vec<Arc<str>>>,
}

impl Operator for FastProjectionOperator {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let chunk = chunk.with_data(chunk.data.select(self.columns.as_slice())?);
        Ok(OperatorResult::Finished(chunk))
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub(crate) struct ProjectionOperator {
    pub(crate) exprs: Vec<Arc<dyn PhysicalPipedExpr>>,
}

impl Operator for ProjectionOperator {
    fn execute(
        &mut self,
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
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub(crate) struct HstackOperator {
    pub(crate) exprs: Vec<Arc<dyn PhysicalPipedExpr>>,
}

impl Operator for HstackOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        let projected = self
            .exprs
            .iter()
            .map(|e| e.evaluate(chunk, context.execution_state.as_ref()))
            .collect::<PolarsResult<Vec<_>>>()?;

        let df = chunk.data.hstack(&projected)?;

        let chunk = chunk.with_data(df);
        Ok(OperatorResult::Finished(chunk))
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
}
