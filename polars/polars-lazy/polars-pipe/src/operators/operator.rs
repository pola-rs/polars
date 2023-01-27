use super::*;

pub enum OperatorResult {
    NeedsNewData,
    // needs to be called again with same chunk.
    HaveMoreOutPut(DataChunk),
    Finished(DataChunk),
}

pub trait Operator: Send + Sync {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult>;

    fn split(&self, thread_no: usize) -> Box<dyn Operator>;

    fn fmt(&self) -> &str;
}
