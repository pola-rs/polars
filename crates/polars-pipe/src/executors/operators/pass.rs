use polars_core::error::PolarsResult;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

/// Simply pass through the chunks
pub struct Pass {
    name: &'static str,
}

impl Pass {
    pub(crate) fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl Operator for Pass {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        Ok(OperatorResult::Finished(chunk.clone()))
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(Self { name: self.name })
    }

    fn fmt(&self) -> &str {
        self.name
    }
}
