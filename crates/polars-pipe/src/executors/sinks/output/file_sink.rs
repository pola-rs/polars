use std::any::Any;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use polars_core::prelude::*;

use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

pub(super) trait SinkWriter {
    fn _write_chunk(&mut self, chunk: DataChunk) -> PolarsResult<()> {
        self._write_batch(&chunk.data)
    }

    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()>;

    fn _finish(&mut self) -> PolarsResult<()>;
}

impl SinkWriter for Box<dyn SinkWriter + Send> {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        self.as_mut()._write_batch(df)
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        self.as_mut()._finish()
    }
}

impl SinkWriter for Vec<DataChunk> {
    fn _write_chunk(&mut self, chunk: DataChunk) -> PolarsResult<()> {
        self.push(chunk);
        Ok(())
    }

    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        panic!("Shouldn't be called");
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        Ok(())
    }
}

/// Write DataChunks in sufficiently large chunks that we don't suffer from
/// overhead of many small writes.
#[derive(Clone)]
pub(crate) struct BufferedWriter<SW> {
    current_chunk: Option<DataChunk>,
    /// Have we vstack()ed on to the current chunk?
    stacked: bool,
    writer: SW,
}

impl<SW: SinkWriter> BufferedWriter<SW> {
    /// Create a new instance.
    pub fn new(writer: SW) -> Self {
        Self {
            current_chunk: None,
            stacked: false,
            writer,
        }
    }

    /// Write (or vstack) another chunk.
    pub fn write(&mut self, next_chunk: DataChunk) {
        // If the next chunk is too large, we probably don't want make copies of
        // it when we do as_single_chunk() in flush(), so we flush in advance.
        if self.current_chunk.is_some() && next_chunk.data.estimated_size() > 10 * 1024 * 1024 {
            self.flush();
        }

        if let Some(ref mut current_chunk) = self.current_chunk {
            current_chunk
                .data
                .vstack_mut(&next_chunk.data)
                .expect("These are chunks from the same dataframe");
            self.stacked = true;
        } else {
            self.current_chunk = Some(next_chunk);
        };
        // 4 MB was chosen based on some empirical experiments that showed it to
        // be decently faster than lower or higher values, and it's small enough
        // it won't impact memory usage significantly.
        if self.current_chunk.as_ref().unwrap().data.estimated_size() > 4 * 1024 * 1024 {
            self.flush();
        }
    }

    /// Do the actual write of any buffered data.
    pub fn flush(&mut self) {
        if let Some(mut current_chunk) = std::mem::take(&mut self.current_chunk) {
            // If we've stacked multiple small batches we want to make the data
            // contiguous.
            if self.stacked {
                current_chunk.data.as_single_chunk();
            }
            self.writer._write_chunk(current_chunk).unwrap();
            self.current_chunk = None;
            self.stacked = false;
        }
    }

    /// Get a reference to the underlying SinkWriter.
    pub fn underlying(&mut self) -> &mut SW {
        &mut self.writer
    }

}

pub(super) fn init_writer_thread(
    receiver: Receiver<Option<DataChunk>>,
    writer: Box<dyn SinkWriter + Send>,
    maintain_order: bool,
    // this is used to determine when a batch of chunks should be written to disk
    // all chunks per push should be collected to determine in which order they should
    // be written
    morsels_per_sink: usize,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        // keep chunks around until all chunks per sink are written
        // then we write them all at once.
        let mut chunks = Vec::with_capacity(morsels_per_sink);
        let mut buffered_writer: BufferedWriter<Box<dyn SinkWriter + Send>> =
            BufferedWriter::new(writer);

        while let Ok(chunk) = receiver.recv() {
            // `last_write` indicates if all chunks are processed, e.g. this is the last write.
            // this is when `write_chunks` is called with `None`.
            let last_write = if let Some(chunk) = chunk {
                chunks.push(chunk);
                false
            } else {
                true
            };

            if chunks.len() == morsels_per_sink || last_write {
                if maintain_order {
                    chunks.sort_by_key(|chunk| chunk.chunk_index);
                }

                for chunk in chunks.drain(0..) {
                    buffered_writer.write(chunk);
                }

                if last_write {
                    buffered_writer.flush();
                    buffered_writer.underlying()._finish().unwrap();
                    return;
                }
            }
        }
    })
}

// Ensure the data is return in the order it was streamed
#[derive(Clone)]
pub struct FilesSink {
    pub(crate) sender: Sender<Option<DataChunk>>,
    pub(crate) io_thread_handle: Arc<Option<JoinHandle<()>>>,
}

impl Sink for FilesSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // don't add empty dataframes
        if chunk.data.height() > 0 {
            self.sender.send(Some(chunk)).unwrap();
        };
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, _other: &mut dyn Sink) {
        // already synchronized
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }
    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        // `None` indicates that we can flush all remaining chunks.
        self.sender.send(None).unwrap();

        // wait until all files written
        // some unwrap/mut kung-fu to get a hold of `self`
        Arc::get_mut(&mut self.io_thread_handle)
            .unwrap()
            .take()
            .unwrap()
            .join()
            .unwrap();

        // return a dummy dataframe;
        Ok(FinalizedSink::Finished(Default::default()))
    }
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    fn fmt(&self) -> &str {
        "parquet_sink"
    }
}
