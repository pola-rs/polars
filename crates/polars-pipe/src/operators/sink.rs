use std::any::Any;
use std::fmt::{Debug, Formatter};

use polars_utils::arena::Node;

use super::*;

#[derive(Debug)]
pub enum SinkResult {
    Finished,
    CanHaveMoreInput,
}

pub enum FinalizedSink {
    Finished(DataFrame),
    Operator,
    Source(Box<dyn Source>),
}

impl Debug for FinalizedSink {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FinalizedSink::Finished(_) => "finished",
            FinalizedSink::Operator => "operator",
            FinalizedSink::Source(_) => "source",
        };
        write!(f, "{s}")
    }
}

pub trait Sink: Send + Sync {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult>;

    fn combine(&mut self, other: &mut dyn Sink);

    fn split(&self, thread_no: usize) -> Box<dyn Sink>;

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink>;

    fn as_any(&mut self) -> &mut dyn Any;

    fn fmt(&self) -> &str;

    fn is_join_build(&self) -> bool {
        false
    }

    // Only implemented for Join sinks
    fn node(&self) -> Node {
        unimplemented!()
    }
}
