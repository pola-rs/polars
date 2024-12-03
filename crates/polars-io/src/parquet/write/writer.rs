use std::io::Write;
use std::sync::Mutex;

use arrow::datatypes::PhysicalType;
use polars_core::prelude::*;
use polars_parquet::write::{
    to_parquet_schema, transverse, CompressionOptions, Encoding, FileWriter, SchemaDescriptor,
    StatisticsOptions, Version, WriteOptions,
};
use polars_utils::idx_vec::UnitVec;

use super::batched_writer::BatchedWriter;
use super::options::{MaterializedSortingColumns, MetadataOptions, ParquetCompression, SortingColumnBehavior};
use super::ParquetWriteOptions;
use crate::prelude::chunk_df_for_writing;
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

pub enum SortingColumns {
    None,
    All(SortingColumnBehavior),
    Fields(PlHashMap<PlSmallStr, SortingColumns>),
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

    sorting_columns: SortingColumns,

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
            compression: ParquetCompression::default().into(),
            statistics: StatisticsOptions::default(),
            row_group_size: None,
            data_page_size: None,
            sorting_columns: SortingColumns::None,
            parallel: true,
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

    /// Set the `SortingColumn`
    pub fn with_sorting_columns(mut self, sorting_columns: SortingColumns) -> Self {
        self.sorting_columns = sorting_columns;
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
        let metadata_options = self.materialize_md_options(&parquet_schema)?;
        let options = self.materialize_options();
        let writer = Mutex::new(FileWriter::try_new(
            self.writer,
            schema,
            options,
        )?);

        Ok(BatchedWriter {
            writer,
            parquet_schema,
            encodings,
            options,
            metadata_options,
            parallel: self.parallel,
        })
    }

    fn materialize_md_options(
        &self,
        parquet_schema: &SchemaDescriptor,
    ) -> PolarsResult<MetadataOptions> {
        let sorting_columns = match &self.sorting_columns {
            SortingColumns::None => MaterializedSortingColumns::All(SortingColumnBehavior::Preserve { force: false }),
            SortingColumns::All(behavior) => MaterializedSortingColumns::All(*behavior),
            SortingColumns::Fields(fields) => {
                let mut col_idx_lookup = PlHashMap::with_capacity(parquet_schema.columns().len());
                for (i, col_descriptor) in parquet_schema.columns().iter().enumerate() {
                    col_idx_lookup.insert(col_descriptor.path_in_schema.as_slice(), i as i32);
                }

                let mut sorting_columns = Vec::new();
                let mut stack = vec![(UnitVec::default(), fields)];

                loop { 
                    let Some((path, fields)) = stack.pop() else {
                        break;
                    };

                    for (name, sc) in fields.iter() {
                        let mut field_path = path.clone();
                        field_path.push(name.clone());

                        let col_idx = col_idx_lookup
                            .get(field_path.as_slice())
                            .ok_or_else(|| polars_err!(col_not_found = path.as_slice().join(" ")))?;

                        match sc {
                            SortingColumns::None => sorting_columns.push((*col_idx, SortingColumnBehavior::default())),
                            SortingColumns::All(sorting_column_behavior) => sorting_columns.push((*col_idx, *sorting_column_behavior)),
                            SortingColumns::Fields(fields) => stack.push((field_path, fields)),
                        }
                    }
                }
                
                sorting_columns.sort_unstable_by_key(|(col_idx, _)| *col_idx);

                MaterializedSortingColumns::PerLeaf(sorting_columns)
            }
        };

        Ok(MetadataOptions { sorting_columns })
    }

    fn materialize_options(&self) -> WriteOptions {
        WriteOptions {
            statistics: self.statistics,
            compression: self.compression,
            version: Version::V1,
            data_page_size: self.data_page_size,
        }
    }

    /// Write the given DataFrame in the writer `W`. Returns the total size of the file.
    pub fn finish(self, df: &mut DataFrame) -> PolarsResult<u64> {
        let chunked_df = chunk_df_for_writing(df, self.row_group_size.unwrap_or(512 * 512))?;
        let mut batched = self.batched(&chunked_df.schema())?;
        batched.write_batch(&chunked_df)?;
        batched.finish()
    }
}

fn get_encodings(schema: &ArrowSchema) -> Vec<Vec<Encoding>> {
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
        PhysicalType::Boolean => Encoding::Rle,
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
