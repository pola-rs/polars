use std::any::Any;
use std::sync::atomic::{AtomicU16, Ordering};

use polars_core::error::PolarsResult;

use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult,
};
use crate::pipeline::morsels_per_sink;

// Ensure the data is return in the order it was streamed
pub struct ParquetSink {
    chunks: Vec<DataChunk>,
    morsels_per_sink: u16,
    morsels_processed: AtomicU16,
}

impl Clone for ParquetSink {
    fn clone(&self) -> Self {
        Self {
            chunks: self.chunks.clone(),
            morsels_per_sink: self.morsels_per_sink,
            morsels_processed: AtomicU16::new(self.morsels_processed.load(Ordering::Acquire)),
        }
    }
}

impl ParquetSink {
    pub fn new() -> Self {
        let morsels_per_sink = morsels_per_sink() as u16;
        ParquetSink {
            chunks: vec![],
            morsels_per_sink,
            morsels_processed: AtomicU16::new(0),
        }
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
