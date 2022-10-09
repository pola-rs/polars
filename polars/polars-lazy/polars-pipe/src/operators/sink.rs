use std::any::Any;

use super::*;

#[derive(Debug)]
pub enum SinkResult {
    Finished,
    CanHaveMoreInput,
}

pub trait Sink: Send + Sync {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult>;

    fn combine(&mut self, other: Box<dyn Sink>);

    fn split(&self, thread_no: usize) -> Box<dyn Sink>;

    fn finalize(&mut self) -> PolarsResult<DataFrame>;

    fn as_any(&self) -> &dyn Any;
}
