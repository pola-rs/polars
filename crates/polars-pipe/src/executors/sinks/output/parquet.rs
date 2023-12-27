use std::path::Path;

use crossbeam_channel::bounded;
use polars_core::prelude::*;
use polars_io::parquet::ParquetWriter;
use polars_plan::prelude::ParquetWriteOptions;

use crate::executors::sinks::output::file_sink::{init_writer_thread, FilesSink, SinkWriter};
use crate::pipeline::morsels_per_sink;

pub struct ParquetSink {}
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
            .with_data_page_size(options.data_pagesize_limit)
            .with_statistics(options.statistics)
            .with_row_group_size(options.row_group_size)
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
pub struct ParquetCloudSink {}
#[cfg(feature = "cloud")]
impl ParquetCloudSink {
    #[allow(clippy::new_ret_no_self)]
    #[tokio::main(flavor = "current_thread")]
    pub async fn new(
        uri: &str,
        cloud_options: Option<&polars_io::cloud::CloudOptions>,
        parquet_options: ParquetWriteOptions,
        schema: &Schema,
    ) -> PolarsResult<FilesSink> {
        let cloud_writer = polars_io::cloud::CloudWriter::new(uri, cloud_options).await?;
        let writer = ParquetWriter::new(cloud_writer)
            .with_compression(parquet_options.compression)
            .with_data_page_size(parquet_options.data_pagesize_limit)
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
            parquet_options.maintain_order,
            morsels_per_sink,
        )));

        Ok(FilesSink {
            sender,
            io_thread_handle,
        })
    }
}

impl<W: std::io::Write> SinkWriter for polars_io::parquet::BatchedWriter<W> {
    fn _write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        self.write_batch(df)
    }

    fn _finish(&mut self) -> PolarsResult<()> {
        self.finish()?;
        Ok(())
    }
}
