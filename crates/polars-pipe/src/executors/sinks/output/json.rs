use std::path::Path;

use crossbeam_channel::bounded;
use polars_core::prelude::*;
use polars_io::json::{BatchedWriter, JsonWriterOptions};

use crate::executors::sinks::output::file_sink::{init_writer_thread, FilesSink, SinkWriter};
use crate::pipeline::morsels_per_sink;

impl SinkWriter for BatchedWriter<std::fs::File> {
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
        options: JsonWriterOptions,
        _schema: &Schema,
    ) -> PolarsResult<FilesSink> {
        let file = std::fs::File::create(path)?;
        let writer = BatchedWriter::new(file);

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
