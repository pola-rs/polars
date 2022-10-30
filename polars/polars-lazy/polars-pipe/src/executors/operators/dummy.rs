use polars_core::error::PolarsResult;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Default)]
pub struct Dummy {}

impl Operator for Dummy {
    fn execute(
        &self,
        _context: &PExecutionContext,
        _chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        panic!("dummy should be replaced")
    }
}
