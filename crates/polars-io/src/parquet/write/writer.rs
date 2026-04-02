use std::io::Write;
use std::sync::Mutex;

use polars_buffer::Buffer;
use polars_core::frame::chunk_df_for_writing;
use polars_core::prelude::*;
use polars_parquet::write::{
    CompressionOptions, Encoding, FileWriter, StatisticsOptions, Version, WriteOptions,
    get_dtype_encoding, to_parquet_schema,
};

use super::batched_writer::BatchedWriter;
use super::options::ParquetCompression;
use super::{KeyValueMetadata, ParquetWriteOptions};
use crate::shared::schema_to_arrow_checked;

impl ParquetWriteOptions {
    pub fn to_writer<F>(&self, f: F) -> ParquetWriter<F>
    where
        F: Write,
    {
        ParquetWriter::new(f)
            .with_compression(self.compression)
            .with_statistics(self.statistics)
            .with_row_group_size(self.row_group_size)
            .with_data_page_size(self.data_page_size)
            .with_key_value_metadata(self.key_value_metadata.clone())
    }
}

/// Write a DataFrame to Parquet format.
#[must_use]
pub struct ParquetWriter<W> {
    writer: W,
    /// Data page compression
    compression: CompressionOptions,
    /// Compute and write column statistics.
    statistics: StatisticsOptions,
    /// if `None` will be 512^2 rows
    row_group_size: Option<usize>,
    /// if `None` will be 1024^2 bytes
    data_page_size: Option<usize>,
    /// Serialize columns in parallel
    parallel: bool,
    /// Custom file-level key value metadata
    key_value_metadata: Option<KeyValueMetadata>,
    /// Context info for the Parquet file being written.
    context_info: Option<PlHashMap<String, String>>,
}

impl<W> ParquetWriter<W>
where
    W: Write,
{
    /// Create a new writer
    pub fn new(writer: W) -> Self
    where
        W: Write,
    {
        ParquetWriter {
            writer,
            compression: ParquetCompression::default().into(),
            statistics: StatisticsOptions::default(),
            row_group_size: None,
            data_page_size: None,
            parallel: true,
            key_value_metadata: None,
            context_info: None,
        }
    }

    /// Set the compression used. Defaults to `Zstd`.
    ///
    /// The default compression `Zstd` has very good performance, but may not yet been supported
    /// by older readers. If you want more compatibility guarantees, consider using `Snappy`.
    pub fn with_compression(mut self, compression: ParquetCompression) -> Self {
        self.compression = compression.into();
        self
    }

    /// Compute and write statistic
    pub fn with_statistics(mut self, statistics: StatisticsOptions) -> Self {
        self.statistics = statistics;
        self
    }

    /// Set the row group size (in number of rows) during writing. This can reduce memory pressure and improve
    /// writing performance.
    pub fn with_row_group_size(mut self, size: Option<usize>) -> Self {
        self.row_group_size = size;
        self
    }

    /// Sets the maximum bytes size of a data page. If `None` will be 1024^2 bytes.
    pub fn with_data_page_size(mut self, limit: Option<usize>) -> Self {
        self.data_page_size = limit;
        self
    }

    /// Serialize columns in parallel
    pub fn set_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set custom file-level key value metadata for the Parquet file
    pub fn with_key_value_metadata(mut self, key_value_metadata: Option<KeyValueMetadata>) -> Self {
        self.key_value_metadata = key_value_metadata;
        self
    }

    /// Set context information for the writer
    pub fn with_context_info(mut self, context_info: Option<PlHashMap<String, String>>) -> Self {
        self.context_info = context_info;
        self
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let schema = schema_to_arrow_checked(schema, CompatLevel::newest(), "parquet")?;
        let parquet_schema = to_parquet_schema(&schema)?;
        let encodings = get_encodings(&schema);
        let options = self.materialize_options();
        let writer = Mutex::new(FileWriter::try_new(self.writer, schema, options)?);

        Ok(BatchedWriter {
            writer,
            parquet_schema,
            encodings,
            options,
            parallel: self.parallel,
            key_value_metadata: self.key_value_metadata,
        })
    }

    fn materialize_options(&self) -> WriteOptions {
        WriteOptions {
            statistics: self.statistics,
            compression: self.compression,
            version: Version::V1,
            data_page_size: self.data_page_size,
        }
    }

    /// Write the given DataFrame in the writer `W`.
    /// Returns the total size of the file.
    pub fn finish(self, df: &mut DataFrame) -> PolarsResult<u64> {
        let chunked_df = chunk_df_for_writing(df, self.row_group_size.unwrap_or(512 * 512))?;
        let mut batched = self.batched(chunked_df.schema())?;
        batched.write_batch(&chunked_df)?;
        batched.finish()
    }
}

pub fn get_encodings(schema: &ArrowSchema) -> Buffer<Vec<Encoding>> {
    schema
        .iter_values()
        .map(|f| get_dtype_encoding(&f.dtype))
        .collect()
}
