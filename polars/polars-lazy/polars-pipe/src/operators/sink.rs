use std::any::Any;

use super::*;

#[derive(Debug)]
pub enum SinkResult {
    Finished,
    CanHaveMoreInput,
}

pub enum FinalizedSink {
    Finished(DataFrame),
    Operator(Box<dyn Operator>)
}

impl FinalizedSink {
    pub(crate) fn unwrap(self) -> DataFrame {
        match self {
            FinalizedSink::Finished(df) => df,
            _ => panic!("not a df")
        }
    }
}

pub trait Sink: Send + Sync {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult>;

    fn combine(&mut self, other: Box<dyn Sink>);

    fn split(&self, thread_no: usize) -> Box<dyn Sink>;

    fn finalize(&mut self) -> PolarsResult<FinalizedSink>;

    fn as_any(&mut self) -> &mut dyn Any;

    fn into_source(&mut self) -> PolarsResult<Option<Box<dyn Source>>> {
        Ok(None)
    }
}
