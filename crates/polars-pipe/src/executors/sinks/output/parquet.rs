use std::any::Any;
use std::path::Path;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver, Sender};
use polars_core::prelude::*;
use polars_io::parquet::write::{
    BatchedWriter, ParquetWriteOptions, ParquetWriter, RowGroupIterColumns,
};

use crate::executors::sinks::output::file_sink::{init_writer_thread, FilesSink, SinkWriter};
use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};
use crate::pipeline::morsels_per_sink;

type RowGroups = Vec<RowGroupIterColumns<'static, PolarsError>>;

pub(super) fn init_row_group_writer_thread(
    receiver: Receiver<Option<(IdxSize, RowGroups)>>,
    writer: Arc<BatchedWriter<std::fs::File>>,
    // this is used to determine when a batch of chunks should be written to disk
    // all chunks per push should be collected to determine in which order they should
    // be written
    morsels_per_sink: usize,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        // keep chunks around until all chunks per sink are written
        // then we write them all at once.
        let mut batched = Vec::with_capacity(morsels_per_sink);
        while let Ok(rgs) = receiver.recv() {
            // `last_write` indicates if all chunks are processed, e.g. this is the last write.
            // this is when `write_chunks` is called with `None`.
            let last_write = if let Some(rgs) = rgs {
                batched.push(rgs);
                false
            } else {
                true
            };

            if batched.len() == morsels_per_sink || last_write {
                batched.sort_by_key(|chunk| chunk.0);

                for (_, rg) in batched.drain(0..) {
                    writer.write_row_groups(rg).unwrap()
                }
            }
            if last_write {
                writer.finish().unwrap();
                return;
            }
        }
    })
}

#[derive(Clone)]
pub struct ParquetSink {
    writer: Arc<BatchedWriter<std::fs::File>>,
    io_thread_handle: Arc<Option<JoinHandle<()>>>,
    sender: Sender<Option<(IdxSize, RowGroups)>>,
}
impl ParquetSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(path: &Path, options: ParquetWriteOptions, schema: &Schema) -> PolarsResult<Self> {
        let file = std::fs::File::create(path)?;
        let writer = ParquetWriter::new(file)
            .with_compression(options.compression)
            .with_data_page_size(options.data_page_size)
            .with_statistics(options.statistics)
            .with_row_group_size(options.row_group_size)
            // This is important! Otherwise we will deadlock
            // See: #7074
            .set_parallel(false)
            .batched(schema)?;

        let writer = Arc::new(writer);
        let morsels_per_sink = morsels_per_sink();

        let backpressure = morsels_per_sink * 4;
        let (sender, receiver) = bounded(backpressure);

        let io_thread_handle = Arc::new(Some(init_row_group_writer_thread(
            receiver,
            writer.clone(),
            morsels_per_sink,
        )));

        Ok(Self {
            writer,
            io_thread_handle,
            sender,
        })
    }
}

impl Sink for ParquetSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // Encode and compress row-groups on every thread.
        let row_groups = self
            .writer
            .encode_and_compress(&chunk.data)
            .collect::<PolarsResult<Vec<_>>>()?;
        // Only then send the compressed pages to the writer.
        self.sender
            .send(Some((chunk.chunk_index, row_groups)))
            .unwrap();
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, _other: &mut dyn Sink) {
        // Nothing to do
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

#[cfg(feature = "cloud")]
pub struct ParquetCloudSink {}
#[cfg(feature = "cloud")]
impl ParquetCloudSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        uri: &str,
        cloud_options: Option<&polars_io::cloud::CloudOptions>,
        parquet_options: ParquetWriteOptions,
        schema: &Schema,
    ) -> PolarsResult<FilesSink> {
        polars_io::pl_async::get_runtime().block_on_potential_spawn(async {
            let cloud_writer = polars_io::cloud::CloudWriter::new(uri, cloud_options).await?;
            let writer = ParquetWriter::new(cloud_writer)
                .with_compression(parquet_options.compression)
                .with_data_page_size(parquet_options.data_page_size)
                .with_statistics(parquet_options.statistics)
                .with_row_group_size(parquet_options.row_group_size)
                // This is important! Otherwise we will deadlock
                // See: #7074
                .set_parallel(false)
                .batched(schema)?;

            let writer = Box::new(writer) as Box<dyn SinkWriter + Send>;

            let morsels_per_sink = morsels_per_sink();
            let backpressure = morsels_per_sink * 2;
            let (sender, receiver) = bounded(backpressure);

            let io_thread_handle = Arc::new(Some(init_writer_thread(
                receiver,
                writer,
                true,
                morsels_per_sink,
            )));

            Ok(FilesSink {
                sender,
                io_thread_handle,
            })
        })
    }
}

impl<W: std::io::Write> SinkWriter for polars_io::parquet::write::BatchedWriter<W> {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        self.write_batch(df)
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        self.finish()?;
        Ok(())
    }
}
