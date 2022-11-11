use std::any::Any;

use polars_core::error::PolarsResult;

use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult,
};

// Ensure the data is return in the order it was streamed
#[derive(Clone)]
pub struct ParquetSink {
    chunks: Vec<DataChunk>,
}

impl ParquetSink {
    pub fn new() -> Self {
        ParquetSink { chunks: vec![] }
    }

    fn sort(&mut self) {
        self.chunks.sort_unstable_by_key(|chunk| chunk.chunk_index);
    }
}

impl Sink for ParquetSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // don't add empty dataframes
        if chunk.data.height() > 0 || self.chunks.is_empty() {
            self.chunks.push(chunk);
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, mut other: Box<dyn Sink>) {
        let other = other.as_any().downcast_ref::<ParquetSink>().unwrap();
        self.chunks.extend_from_slice(&other.chunks);
        self.sort();
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }
    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        self.sort();
        let chunks = std::mem::take(&mut self.chunks);
        Ok(FinalizedSink::Finished(chunks_to_df_unchecked(chunks)))
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
