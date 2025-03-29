use std::io::Write;
use std::sync::Mutex;

use arrow::datatypes::PhysicalType;
use polars_core::frame::chunk_df_for_writing;
use polars_core::prelude::*;
use polars_parquet::write::{
    Encoding, FileWriter, StatisticsOptions, Version, WriteOptions, to_parquet_schema, transverse,
};

use super::ParquetWriteOptions;
use super::batched_writer::BatchedWriter;
use super::options::ParquetCompression;
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
    }
}

/// Write a DataFrame to Parquet format.
#[must_use]
pub struct ParquetWriter<W> {
    writer: W,
    options: WriteOptions,
    /// if `None` will be 512^2 rows
    row_group_size: Option<usize>,
    /// Serialize columns in parallel
    parallel: bool,
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
            options: WriteOptions {
                compression: ParquetCompression::default().into(),
                page_index: false,
                version: Version::V1,
                statistics: StatisticsOptions::default(),
                data_page_size: None,
            },
            row_group_size: None,
            parallel: true,
        }
    }

    /// Set the compression used. Defaults to `Zstd`.
    ///
    /// The default compression `Zstd` has very good performance, but may not yet been supported
    /// by older readers. If you want more compatibility guarantees, consider using `Snappy`.
    pub fn with_compression(mut self, compression: ParquetCompression) -> Self {
        self.options.compression = compression.into();
        self
    }

    /// Compute and write statistic
    pub fn with_statistics(mut self, statistics: StatisticsOptions) -> Self {
        self.options.statistics = statistics;
        self
    }

    /// Compute and write statistic
    pub fn with_page_index(mut self, page_index: bool) -> Self {
        self.options.page_index = page_index;
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
        self.options.data_page_size = limit;
        self
    }

    /// Serialize columns in parallel
    pub fn set_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let schema = schema_to_arrow_checked(schema, CompatLevel::newest(), "parquet")?;
        let parquet_schema = to_parquet_schema(&schema)?;
        let encodings = get_encodings(&schema);
        let options = self.options;
        let writer = Mutex::new(FileWriter::try_new(self.writer, schema, options)?);

        Ok(BatchedWriter {
            writer,
            parquet_schema,
            encodings,
            options,
            parallel: self.parallel,
        })
    }

    /// Write the given DataFrame in the writer `W`. Returns the total size of the file.
    pub fn finish(self, df: &mut DataFrame) -> PolarsResult<u64> {
        let chunked_df = chunk_df_for_writing(df, self.row_group_size.unwrap_or(512 * 512))?;
        let mut batched = self.batched(chunked_df.schema())?;
        batched.write_batch(&chunked_df)?;
        batched.finish()
    }
}

pub fn get_encodings(schema: &ArrowSchema) -> Vec<Vec<Encoding>> {
    schema
        .iter_values()
        .map(|f| transverse(&f.dtype, encoding_map))
        .collect()
}

/// Declare encodings
fn encoding_map(dtype: &ArrowDataType) -> Encoding {
    match dtype.to_physical_type() {
        PhysicalType::Dictionary(_)
        | PhysicalType::LargeBinary
        | PhysicalType::LargeUtf8
        | PhysicalType::Utf8View
        | PhysicalType::BinaryView => Encoding::RleDictionary,
        PhysicalType::Primitive(dt) => {
            use arrow::types::PrimitiveType::*;
            match dt {
                Float32 | Float64 | Float16 => Encoding::Plain,
                _ => Encoding::RleDictionary,
            }
        },
        // remaining is plain
        _ => Encoding::Plain,
    }
}
