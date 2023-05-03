use polars_core::error::PolarsResult;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

/// Simply pass through the chunks
#[derive(Default)]
pub struct Pass {}

impl Operator for Pass {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
       chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        Ok(OperatorResult::Finished(chunk.clone()))
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(Self {})
    }

    fn fmt(&self) -> &str {
        "pass"
    }
}
