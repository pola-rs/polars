use super::*;

pub enum OperatorResult {
    NeedMoreInput(Option<DataChunk>),
    HaveMoreOutPut(Option<DataChunk>),
    Finished(Option<DataChunk>),
}

pub trait Operator {
    fn execute(
        &self,
        context: &PExecutionContext,
        chunk: DataChunk,
    ) -> PolarsResult<OperatorResult>;
}
