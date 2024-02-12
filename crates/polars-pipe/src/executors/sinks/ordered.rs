use std::any::Any;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;

use crate::operators::{
    estimated_chunks, DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult,
    StreamingVstacker,
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
            return Ok(FinalizedSink::Finished(DataFrame::from(
                self.schema.as_ref(),
            )));
        }
        self.sort();

        let chunks = std::mem::take(&mut self.chunks);
        let mut combiner = StreamingVstacker::default();
        let mut frames_iterator = chunks
            .into_iter()
            .flat_map(|c| combiner.add(c.data))
            .map(|mut df| {
                // The dataframe may only be a single, large chunk, in
                // which case we don't want to bother with copying it...
                if estimated_chunks(&df) > 1 {
                    df.as_single_chunk_par();
                }
                df
            })
            .peekable();
        let result = if frames_iterator.peek().is_some() {
            let mut result = accumulate_dataframes_vertical_unchecked(frames_iterator);
            if let Some(mut df) = combiner.finish() {
                if estimated_chunks(&df) > 1 {
                    df.as_single_chunk_par();
                }
                let _ = result.vstack_mut(&df);
            }
            result
        } else {
            combiner.finish().unwrap()
        };
        Ok(FinalizedSink::Finished(result))
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn fmt(&self) -> &str {
        "ordered_sink"
    }
}
