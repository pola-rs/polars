use std::any::Any;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use polars_core::prelude::*;

use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

pub(super) trait SinkWriter {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()>;
    fn _finish(&mut self) -> PolarsResult<()>;
}

/// Write DataChunks in sufficiently large chunks that we don't suffer from
/// overhead of many small writes.
struct BufferedWriter {
    current_frame: Option<DataFrame>,
    /// Have we vstack()ed on to the current chunk?
    stacked: bool,
    writer: Box<dyn SinkWriter + Send>,
}

impl BufferedWriter {
    /// Create a new instance.
    fn new(writer: Box<dyn SinkWriter + Send>) -> Self {
        Self {
            current_frame: None,
            stacked: false,
            writer,
        }
    }

    /// Write (or vstack) another chunk.
    fn write(&mut self, next_chunk: DataChunk) {
        // If the next chunk is too large, we probably don't want make copies of
        // it when we do as_single_chunk() in _flush(), so we flush in advance.
        if self.current_frame.is_some() && next_chunk.data.estimated_size() > 10 * 1024 * 1024 {
            self._flush();
        }

        if let Some(ref mut current_frame) = self.current_frame {
            current_frame
                .vstack_mut(&next_chunk.data)
                .expect("These are chunks from the same dataframe");
            self.stacked = true;
        } else {
            self.current_frame = Some(next_chunk.data);
        };
        // 4 MB was chosen based on some empirical experiments that showed it to
        // be decently faster than lower or higher values, and it's small enough
        // it won't impact memory usage significantly.
        if self.current_frame.as_ref().unwrap().estimated_size() > 4 * 1024 * 1024 {
            self._flush();
        }
    }

    /// Finish writing, flushing remaining data.
    fn finish(mut self) {
        if self.current_frame.is_some() {
            self._flush();
        }
        self.writer._finish().unwrap();
    }

    /// Do the actual write of any buffered data.
    fn _flush(&mut self) {
        let current_frame = self.current_frame.as_mut().unwrap();
        // If we've stacked multiple small batches we want to make the data
        // contiguous.
        if self.stacked {
            current_frame.as_single_chunk();
        }
        self.writer._write_batch(current_frame).unwrap();
        self.current_frame = None;
        self.stacked = false;
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
        let mut buffered_writer = BufferedWriter::new(writer);

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
                    buffered_writer.finish();
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
