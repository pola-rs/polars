use super::*;

pub enum OperatorResult {
    /// needs to be called again with new chunk.
    /// Or in case of `flush` needs to be called again.
    NeedsNewData,
    /// needs to be called again with same chunk.
    HaveMoreOutPut(DataChunk),
    /// this operator is finished
    Finished(DataChunk),
}

pub trait Operator: Send + Sync {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult>;

    fn flush(&mut self) -> PolarsResult<OperatorResult> {
        unimplemented!()
    }

    fn must_flush(&self) -> bool {
        false
    }

    fn split(&self, thread_no: usize) -> Box<dyn Operator>;

    fn fmt(&self) -> &str;
}
