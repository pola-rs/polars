use std::path::Path;

use crossbeam_channel::bounded;
use polars_core::prelude::*;
use polars_io::ipc::IpcWriterOptions;
use polars_io::prelude::*;

use crate::executors::sinks::output::file_sink::{init_writer_thread, FilesSink, SinkWriter};
use crate::pipeline::morsels_per_sink;

pub struct IpcSink {}
impl IpcSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(path: &Path, options: IpcWriterOptions, schema: &Schema) -> PolarsResult<FilesSink> {
        let file = std::fs::File::create(path)?;
        let writer = IpcWriter::new(file)
            .with_compression(options.compression)
            .batched(schema)?;

        let writer = Box::new(writer) as Box<dyn SinkWriter + Send>;

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

#[cfg(feature = "cloud")]
pub struct IpcCloudSink {}
#[cfg(feature = "cloud")]
impl IpcCloudSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        uri: &str,
        cloud_options: Option<&polars_io::cloud::CloudOptions>,
        ipc_options: IpcWriterOptions,
        schema: &Schema,
    ) -> PolarsResult<FilesSink> {
        polars_io::pl_async::get_runtime().block_on_potential_spawn(async {
            let cloud_writer = polars_io::cloud::CloudWriter::new(uri, cloud_options).await?;
            let writer = IpcWriter::new(cloud_writer)
                .with_compression(ipc_options.compression)
                .batched(schema)?;

            let writer = Box::new(writer) as Box<dyn SinkWriter + Send>;

            let morsels_per_sink = morsels_per_sink();
            let backpressure = morsels_per_sink * 2;
            let (sender, receiver) = bounded(backpressure);

            let io_thread_handle = Arc::new(Some(init_writer_thread(
                receiver,
                writer,
                ipc_options.maintain_order,
                morsels_per_sink,
            )));

            Ok(FilesSink {
                sender,
                io_thread_handle,
            })
        })
    }
}

impl<W: std::io::Write> SinkWriter for polars_io::ipc::BatchedWriter<W> {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        self.write_batch(df)
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        self.finish()?;
        Ok(())
    }
}
