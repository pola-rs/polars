use std::any::Any;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_utils::atomic::SyncCounter;

use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult,
};

// Ensure the data is return in the order it was streamed
pub struct SliceSink {
    offset: SyncCounter,
    current_len: SyncCounter,
    len: usize,
    chunks: Arc<Mutex<Vec<DataChunk>>>,
    schema: SchemaRef,
}

impl Clone for SliceSink {
    fn clone(&self) -> Self {
        Self {
            offset: self.offset.clone(),
            current_len: self.current_len.clone(),
            len: self.len,
            chunks: self.chunks.clone(),
            schema: self.schema.clone(),
        }
    }
}

impl SliceSink {
    pub fn new(offset: u64, len: usize, schema: SchemaRef) -> SliceSink {
        let offset = SyncCounter::new(offset as usize);
        SliceSink {
            offset,
            current_len: SyncCounter::new(0),
            len,
            chunks: Default::default(),
            schema,
        }
    }

    fn sort(&mut self) {
        let mut chunks = self.chunks.lock().unwrap();
        chunks.sort_unstable_by_key(|chunk| chunk.chunk_index);
    }
}

impl Sink for SliceSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // there is contention here.

        let height = chunk.data.height();
        let mut chunks = self.chunks.lock().unwrap();
        // don't add empty dataframes
        if height > 0 || chunks.is_empty() {
            // TODO! deal with offset
            // this is a bit harder as the chunks come in randomly

            // we are under a mutex lock here
            // so ordering doesn't seem too important
            let current_offset = self.offset.load(Ordering::Acquire);
            let current_len = self.current_len.fetch_add(height, Ordering::Acquire);

            // always push as they come in random order

            chunks.push(chunk);

            if current_len > (self.len + current_offset) {
                Ok(SinkResult::Finished)
            } else {
                Ok(SinkResult::CanHaveMoreInput)
            }
        } else {
            Ok(SinkResult::CanHaveMoreInput)
        }
    }

    fn combine(&mut self, _other: &mut dyn Sink) {
        // no-op
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        self.sort();
        let chunks = std::mem::take(&mut self.chunks);
        let mut chunks = chunks.lock().unwrap();
        let chunks: Vec<DataChunk> = std::mem::take(chunks.as_mut());
        if chunks.is_empty() {
            return Ok(FinalizedSink::Finished(DataFrame::empty_with_schema(
                &self.schema,
            )));
        }

        let df = chunks_to_df_unchecked(chunks);
        let offset = self.offset.load(Ordering::Acquire) as i64;

        // drop the counters
        unsafe {
            self.offset.manual_drop();
            self.current_len.manual_drop();
        }

        Ok(FinalizedSink::Finished(df.slice(offset, self.len)))
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn fmt(&self) -> &str {
        "slice_sink"
    }
}
