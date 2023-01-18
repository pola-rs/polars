use std::any::Any;
use std::path::Path;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver, Sender};
use polars_core::prelude::*;
#[cfg(feature = "parquet")]
use polars_io::parquet::ParquetWriter;
#[cfg(feature = "ipc")]
use polars_io::prelude::IpcWriter;
#[cfg(feature = "ipc")]
use polars_io::SerWriter;
use polars_plan::prelude::*;

use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};
use crate::pipeline::morsels_per_sink;

#[cfg(any(feature = "parquet", feature = "ipc"))]
trait SinkWriter {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()>;
    fn _finish(&mut self) -> PolarsResult<()>;
}

#[cfg(any(feature = "parquet", feature = "ipc"))]
impl SinkWriter for polars_io::parquet::BatchedWriter<std::fs::File> {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        self.write_batch(df)
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        self.finish()?;
        Ok(())
    }
}

#[cfg(feature = "ipc")]
impl SinkWriter for polars_io::ipc::BatchedWriter<std::fs::File> {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        self.write_batch(df)
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        self.finish()?;
        Ok(())
    }
}

#[cfg(feature = "parquet")]
pub struct ParquetSink {}
#[cfg(feature = "parquet")]
impl ParquetSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        path: &Path,
        options: ParquetWriteOptions,
        schema: &Schema,
    ) -> PolarsResult<FilesSink> {
        let file = std::fs::File::create(path)?;
        let writer = ParquetWriter::new(file)
            .with_compression(options.compression)
            .with_data_pagesize_limit(options.data_pagesize_limit)
            .with_statistics(options.statistics)
            .with_row_group_size(options.row_group_size)
            .batched(schema)?;

        let writer = Box::new(writer) as Box<dyn SinkWriter + Send + Sync>;

        let morsels_per_sink = morsels_per_sink();
        let backpressure = morsels_per_sink * 2;
        let (sender, receiver) = bounded(backpressure);

        let io_thread_handle = Arc::new(Some(init_writer_thread(
            receiver,
            writer,
            options.maintain_order,
            morsels_per_sink,
        )));

        Ok(FilesSink {
            sender,
            io_thread_handle,
        })
    }
}

#[cfg(feature = "ipc")]
pub struct IpcSink {}
#[cfg(feature = "ipc")]
impl IpcSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(path: &Path, options: IpcWriterOptions, schema: &Schema) -> PolarsResult<FilesSink> {
        let file = std::fs::File::create(path)?;
        let writer = IpcWriter::new(file)
            .with_compression(options.compression)
            .batched(schema)?;

        let writer = Box::new(writer) as Box<dyn SinkWriter + Send + Sync>;

        let morsels_per_sink = morsels_per_sink();
        let backpressure = morsels_per_sink * 2;
        let (sender, receiver) = bounded(backpressure);

        let io_thread_handle = Arc::new(Some(init_writer_thread(
            receiver,
            writer,
            options.maintain_order,
            morsels_per_sink,
        )));

        Ok(FilesSink {
            sender,
            io_thread_handle,
        })
    }
}

#[cfg(any(feature = "parquet", feature = "ipc"))]
fn init_writer_thread(
    receiver: Receiver<Option<DataChunk>>,
    mut writer: Box<dyn SinkWriter + Send + Sync>,
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

                for chunk in chunks.iter() {
                    writer._write_batch(&chunk.data).unwrap()
                }
                // all chunks are written remove them
                chunks.clear();

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
#[cfg(any(feature = "parquet", feature = "ipc"))]
pub struct FilesSink {
    sender: Sender<Option<DataChunk>>,
    io_thread_handle: Arc<Option<JoinHandle<()>>>,
}

#[cfg(any(feature = "parquet", feature = "ipc"))]
impl Sink for FilesSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // don't add empty dataframes
        if chunk.data.height() > 0 {
            self.sender.send(Some(chunk)).unwrap();
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
