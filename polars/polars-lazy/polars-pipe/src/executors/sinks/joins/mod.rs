use std::any::Any;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::JoinType::Outer;
use crate::operators::{DataChunk, PExecutionContext, Sink, SinkResult};


#[derive(Default)]
pub struct OuterJoin {
    chunks: Vec<DataChunk>
}

impl Sink for OuterJoin {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        self.chunks.push(chunk);
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, mut other: Box<dyn Sink>) {
        let other = other.as_any().downcast_mut::<Self>().unwrap();
        let other_chunks = std::mem::take(&mut other.chunks);
        self.chunks.extend(other_chunks.into_iter());
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(Self::default())
    }

    fn finalize(&mut self) -> PolarsResult<DataFrame> {
        todo!()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}