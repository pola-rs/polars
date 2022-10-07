use super::*;

pub enum OperatorResult {
    NeedMoreInput,
    HaveMoreOutPut(DataChunk),
    Finished(DataChunk),
}

pub trait Operator: Send + Sync {
    fn execute(
        &self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult>;
}
