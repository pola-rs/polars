use std::any::Any;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;

use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult,
};

// Ensure the data is return in the order it was streamed
#[derive(Clone)]
pub struct OrderedSink {
    chunks: Vec<DataChunk>,
    schema: SchemaRef,
}

impl OrderedSink {
    pub fn new(schema: SchemaRef) -> Self {
        OrderedSink {
            chunks: vec![],
            schema,
        }
    }

    fn sort(&mut self) {
        self.chunks.sort_unstable_by_key(|chunk| chunk.chunk_index);
    }
}

impl Sink for OrderedSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // don't add empty dataframes
        if chunk.data.height() > 0 || self.chunks.is_empty() {
            self.chunks.push(chunk);
        }
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        let other = other.as_any().downcast_ref::<OrderedSink>().unwrap();
        self.chunks.extend_from_slice(&other.chunks);
        self.sort();
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }
    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        if self.chunks.is_empty() {
            return Ok(FinalizedSink::Finished(DataFrame::empty_with_schema(
                &self.schema,
            )));
        }
        self.sort();

        let chunks = std::mem::take(&mut self.chunks);
        Ok(FinalizedSink::Finished(chunks_to_df_unchecked(chunks)))
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn fmt(&self) -> &str {
        "ordered_sink"
    }
}
