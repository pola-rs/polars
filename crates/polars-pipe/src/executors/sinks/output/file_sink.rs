use std::any::Any;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};
use polars_core::prelude::*;

use crate::operators::{
    DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult, StreamingVstacker,
};

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
) -> JoinHandle<PolarsResult<()>> {
    std::thread::spawn(move || -> PolarsResult<()> {
        // keep chunks around until all chunks per sink are written
        // then we write them all at once.
        let mut chunks = Vec::with_capacity(morsels_per_sink);
        let mut vstacker = StreamingVstacker::default();

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
                    for mut df in vstacker.add(chunk.data) {
                        // The dataframe may only be a single, large chunk, in
                        // which case we don't want to bother with copying it...
                        if df.n_chunks() > 1 {
                            df.as_single_chunk();
                        }
                        writer._write_batch(&df)?;
                    }
                }
                // all chunks are written remove them
                chunks.clear();

                if last_write {
                    if let Some(mut df) = vstacker.finish() {
                        if df.n_chunks() > 1 {
                            df.as_single_chunk();
                        }
                        writer._write_batch(&df)?;
                    }
                    writer._finish()?;
                    return Ok(());
                }
            }
        }
        Ok(())
    })
}

// Ensure the data is return in the order it was streamed
#[derive(Clone)]
pub struct FilesSink {
    pub(crate) sender: Sender<Option<DataChunk>>,
    pub(crate) io_thread_handle: Arc<Option<JoinHandle<PolarsResult<()>>>>,
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
            .unwrap()?;

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
