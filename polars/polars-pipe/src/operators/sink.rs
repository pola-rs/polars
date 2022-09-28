use super::*;

pub enum SinkResult {
    Finished,
    NeedMoreInput,
}

pub trait Sink {
    fn sink(context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult>;
}
