use std::any::Any;
use std::path::Path;
use std::sync::Mutex;

use polars_core::prelude::*;
use polars_io::parquet::{BatchedWriter, ParquetWriter};
use polars_plan::prelude::ParquetWriteOptions;
use polars_utils::cell::SyncUnsafeCell;

use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};
use crate::pipeline::morsels_per_sink;

// Ensure the data is return in the order it was streamed
pub struct ParquetSink {
    writer: Arc<SyncUnsafeCell<BatchedWriter<std::fs::File>>>,
    chunks: Arc<Mutex<Vec<DataChunk>>>,
    morsels_per_sink: u16,
    maintain_order: bool,
}

impl Clone for ParquetSink {
    fn clone(&self) -> Self {
        Self {
            writer: self.writer.clone(),
            chunks: self.chunks.clone(),
            morsels_per_sink: self.morsels_per_sink,
            maintain_order: self.maintain_order,
        }
    }
}

impl ParquetSink {
    pub fn new(path: &Path, options: ParquetWriteOptions, schema: &Schema) -> PolarsResult<Self> {
        let file = std::fs::File::create(path)?;
        let writer = ParquetWriter::new(file)
            .with_compression(options.compression)
            .with_data_pagesize_limit(options.data_pagesize_limit)
            .with_statistics(options.statistics)
            .with_row_group_size(options.row_group_size)
            .batched(schema)?;

        let morsels_per_sink = morsels_per_sink() as u16;

        Ok(ParquetSink {
            writer: Arc::new(SyncUnsafeCell::new(writer)),
            chunks: Default::default(),
            morsels_per_sink,
            maintain_order: options.maintain_order,
        })
    }

    // if chunk is `None` we will finalize the writer
    fn write_chunks(&mut self, chunk: Option<DataChunk>) -> PolarsResult<()> {
        let mut chunks = self.chunks.lock().unwrap();
        let last_write = if let Some(chunk) = chunk {
            chunks.push(chunk);
            false
        } else {
            true
        };

        // TODO! speed this up by having a write thread that will make this async
        if chunks.len() as u16 == self.morsels_per_sink || last_write {
            // safety: we hold the mutex lock in chunks
            let writer = unsafe { &mut *(*self.writer).get() };

            if self.maintain_order {
                chunks.sort_by_key(|chunk| chunk.chunk_index);
            }

            for chunk in chunks.iter() {
                writer.write_batch(&chunk.data)?
            }
            // all chunks are written remove them
            chunks.clear();

            if last_write {
                writer.finish()?;
            }
        }
        Ok(())
    }
}

impl Sink for ParquetSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // don't add empty dataframes
        if chunk.data.height() > 0 {
            self.write_chunks(Some(chunk))?;
        };
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, _other: Box<dyn Sink>) {
        // already synchronized
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }
    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        // write remaining chunks
        self.write_chunks(None)?;

        // return a dummy dataframe;
        Ok(FinalizedSink::Finished(Default::default()))
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
