use std::path::Path;

use cloud::CloudOptions;
use crossbeam_channel::bounded;
use file::try_get_writeable;
use polars_core::prelude::*;
use polars_io::ipc::IpcWriterOptions;
use polars_io::prelude::*;

use crate::executors::sinks::output::file_sink::{FilesSink, SinkWriter, init_writer_thread};
use crate::pipeline::morsels_per_sink;

pub struct IpcSink {}
impl IpcSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        path: &Path,
        options: IpcWriterOptions,
        schema: &Schema,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<FilesSink> {
        let writer = IpcWriter::new(try_get_writeable(path.to_str().unwrap(), cloud_options)?)
            .with_compression(options.compression)
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
