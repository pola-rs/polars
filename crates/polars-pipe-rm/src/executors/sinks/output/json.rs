use std::path::Path;

use crossbeam_channel::bounded;
use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::json::{BatchedWriter, JsonWriterOptions};
use polars_io::utils::file::try_get_writeable;

use crate::executors::sinks::output::file_sink::{FilesSink, SinkWriter, init_writer_thread};
use crate::pipeline::morsels_per_sink;

impl<W: std::io::Write> SinkWriter for BatchedWriter<W> {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        self.write_batch(df)
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        Ok(())
    }
}

pub struct JsonSink {}
impl JsonSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        path: &Path,
        _options: JsonWriterOptions,
        _schema: &Schema,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<FilesSink> {
        let writer = BatchedWriter::new(try_get_writeable(path.to_str().unwrap(), cloud_options)?);
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
