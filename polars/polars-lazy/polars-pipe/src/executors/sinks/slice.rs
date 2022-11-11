use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use polars_core::error::PolarsResult;

use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult,
};

// Ensure the data is return in the order it was streamed
#[derive(Clone)]
pub struct SliceSink {
    offset: Arc<AtomicU64>,
    current_len: Arc<AtomicU64>,
    len: usize,
    chunks: Arc<Mutex<Vec<DataChunk>>>,
}

impl SliceSink {
    pub fn new(offset: u64, len: usize) -> SliceSink {
        let offset = Arc::new(AtomicU64::new(offset));
        SliceSink {
            offset,
            current_len: Arc::new(AtomicU64::new(0)),
            len,
            chunks: Default::default(),
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
            let current_offset = self.offset.load(Ordering::Acquire) as usize;
            let current_len = self.current_len.fetch_add(height as u64, Ordering::Acquire) as usize;

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

    fn combine(&mut self, _other: Box<dyn Sink>) {
        // no-op
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        self.sort();
        let chunks = std::mem::take(&mut self.chunks);
        let mut chunks = chunks.lock().unwrap();
        let chunks = std::mem::take(chunks.as_mut());
        let df = chunks_to_df_unchecked(chunks);
        let offset = self.offset.load(Ordering::Acquire) as i64;
        Ok(FinalizedSink::Finished(df.slice(offset, self.len)))
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
