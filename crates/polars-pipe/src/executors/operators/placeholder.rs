use polars_core::error::PolarsResult;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Default)]
pub struct PlaceHolder {}

impl Operator for PlaceHolder {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        _chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        panic!("placeholder should be replaced")
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(Self {})
    }

    fn fmt(&self) -> &str {
        "placeholder"
    }
}
