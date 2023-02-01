use polars_core::error::PolarsResult;
use polars_plan::prelude::*;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub struct FunctionOperator {
    pub(crate) function: FunctionNode,
}

impl Operator for FunctionOperator {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        Ok(OperatorResult::Finished(
            chunk.with_data(self.function.evaluate(chunk.data.clone())?),
        ))
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }

    fn fmt(&self) -> &str {
        "function"
    }
}
