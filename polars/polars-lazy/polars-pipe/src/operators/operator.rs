use super::*;

pub enum OperatorResult {
    NeedMoreInput,
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
}
