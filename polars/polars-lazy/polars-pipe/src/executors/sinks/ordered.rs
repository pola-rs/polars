use std::any::Any;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use crate::operators::{DataChunk, PExecutionContext, Sink, SinkResult};

// Ensure the data is return in the order it was streamed
#[derive(Clone)]
pub struct OrderedSink {
    chunks: Vec<DataChunk>,
}

impl OrderedSink {
    pub fn new() -> Self {
        OrderedSink { chunks: vec![] }
    }

    fn sort(&mut self) {
        self.chunks.sort_unstable_by_key(|chunk| chunk.chunk_index);
    }
}

impl Sink for OrderedSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        self.chunks.push(chunk);
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: Box<dyn Sink>) {
        let other = other.as_any().downcast_ref::<OrderedSink>().unwrap();
        self.chunks.extend_from_slice(&other.chunks);
        self.sort();
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }
    fn finalize(&mut self) -> PolarsResult<DataFrame> {
        self.sort();
        Ok(accumulate_dataframes_vertical_unchecked(
            std::mem::take(&mut self.chunks)
                .into_iter()
                .map(|chunk| chunk.data),
        ))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
