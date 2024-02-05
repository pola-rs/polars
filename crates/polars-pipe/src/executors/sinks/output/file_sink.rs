use std::{any::Any, collections::VecDeque};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use polars_core::prelude::*;

use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

pub(super) trait SinkWriter {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()>;
    fn _finish(&mut self) -> PolarsResult<()>;
}

pub(super) fn init_writer_thread(
    receiver: Receiver<Option<DataChunk>>,
    mut writer: Box<dyn SinkWriter + Send>,
    maintain_order: bool,
    // this is used to determine when a batch of chunks should be written to disk
    // all chunks per push should be collected to determine in which order they should
    // be written
    morsels_per_sink: usize,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        // keep chunks around until all chunks per sink are written
        // then we write them all at once.
        let mut chunks = VecDeque::with_capacity(morsels_per_sink);

        while let Ok(chunk) = receiver.recv() {
            // `last_write` indicates if all chunks are processed, e.g. this is the last write.
            // this is when `write_chunks` is called with `None`.
            let last_write = if let Some(chunk) = chunk {
                chunks.push_back(chunk);
                false
            } else {
                true
            };

            if chunks.len() == morsels_per_sink || last_write {
                if maintain_order {
                    chunks.make_contiguous().sort_by_key(|chunk| chunk.chunk_index);
                }

                // Combine small chunks so we're not doing lots of small writes,
                // which add expensive overhead. TODO might want to disable for
                // some data formats where it might be pointless, e.g. JSON?
                while !chunks.is_empty() {
                    let mut chunk = chunks.pop_front().expect("we checked it's not empty");
                    let mut stacked = false;
                    while chunk.data.estimated_size() < 1024 * 1024 {
                        if let Some(next_chunk) = chunks.pop_front() {
                            chunk.data.vstack_mut(&next_chunk.data).expect("Should be same schema!");
                            stacked = true;
                        } else {
                            break;
                        }
                    }
                    if stacked {
                        chunk.data.as_single_chunk();
                    }
                    writer._write_batch(&chunk.data).unwrap();
                }

                if last_write {
                    writer._finish().unwrap();
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
