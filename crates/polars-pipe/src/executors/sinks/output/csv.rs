use std::path::Path;

use crossbeam_channel::bounded;
use polars_core::prelude::*;
use polars_io::csv::write::{CsvWriter, CsvWriterOptions};
use polars_io::SerWriter;

use crate::executors::sinks::output::file_sink::{init_writer_thread, FilesSink, SinkWriter};
use crate::pipeline::morsels_per_sink;

pub struct CsvSink {}
impl CsvSink {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(path: &Path, options: CsvWriterOptions, schema: &Schema) -> PolarsResult<FilesSink> {
        let file = std::fs::File::create(path)?;
        let writer = CsvWriter::new(file)
            .include_bom(options.include_bom)
            .include_header(options.include_header)
            .with_separator(options.serialize_options.separator)
            .with_line_terminator(options.serialize_options.line_terminator)
            .with_quote_char(options.serialize_options.quote_char)
            .with_batch_size(options.batch_size)
            .with_datetime_format(options.serialize_options.datetime_format)
            .with_date_format(options.serialize_options.date_format)
            .with_time_format(options.serialize_options.time_format)
            .with_float_scientific(options.serialize_options.float_scientific)
            .with_float_precision(options.serialize_options.float_precision)
            .with_null_value(options.serialize_options.null)
            .with_quote_style(options.serialize_options.quote_style)
            .n_threads(1)
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

impl SinkWriter for polars_io::csv::write::BatchedWriter<std::fs::File> {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        self.write_batch(df)
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        self.finish()
    }
}
