use std::io::Write;
use std::sync::Mutex;

use arrow::datatypes::PhysicalType;
use polars_core::frame::chunk_df_for_writing;
use polars_core::prelude::*;
use polars_parquet::write::{
    ChildWriteOptions, ColumnWriteOptions, CompressionOptions, Encoding, FieldWriteOptions,
    FileWriter, KeyValue, ListLikeFieldWriteOptions, StatisticsOptions, StructFieldWriteOptions,
    Version, WriteOptions, to_parquet_schema,
};

use super::batched_writer::BatchedWriter;
use super::options::ParquetCompression;
use super::{KeyValueMetadata, MetadataKeyValue, ParquetFieldOverwrites, ParquetWriteOptions};
use crate::prelude::ChildFieldOverwrites;
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
    field_overwrites: Vec<ParquetFieldOverwrites>,
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
            field_overwrites: Vec::new(),
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
        let column_options = get_column_write_options(&schema, &self.field_overwrites);
        let parquet_schema = to_parquet_schema(&schema, &column_options)?;
        let options = self.materialize_options();
        let writer = Mutex::new(FileWriter::try_new(
            self.writer,
            schema,
            options,
            &column_options,
        )?);

        Ok(BatchedWriter {
            writer,
            parquet_schema,
            column_options,
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

fn convert_metadata(md: &Option<Vec<MetadataKeyValue>>) -> Vec<KeyValue> {
    md.as_ref()
        .map(|metadata| {
            metadata
                .iter()
                .map(|kv| KeyValue {
                    key: kv.key.to_string(),
                    value: kv.value.as_ref().map(|v| v.to_string()),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn to_column_write_options_rec(
    field: &ArrowField,
    overwrites: Option<&ParquetFieldOverwrites>,
) -> ColumnWriteOptions {
    let mut column_options = ColumnWriteOptions {
        field_id: None,
        metadata: Vec::new(),

        // Dummy value.
        children: ChildWriteOptions::Leaf(FieldWriteOptions {
            encoding: Encoding::Plain,
        }),
    };

    if let Some(overwrites) = overwrites {
        column_options.field_id = overwrites.field_id;
        column_options.metadata = convert_metadata(&overwrites.metadata);
    }

    use arrow::datatypes::PhysicalType::*;
    match field.dtype().to_physical_type() {
        Null | Boolean | Primitive(_) | Binary | FixedSizeBinary | LargeBinary | Utf8
        | Dictionary(_) | LargeUtf8 | BinaryView | Utf8View => {
            column_options.children = ChildWriteOptions::Leaf(FieldWriteOptions {
                encoding: encoding_map(field.dtype()),
            });
        },
        List | FixedSizeList | LargeList => {
            let child_overwrites = overwrites.map(|o| match &o.children {
                ChildFieldOverwrites::ListLike(child_overwrites) => child_overwrites.as_ref(),
                _ => unreachable!(),
            });

            let a = field.dtype().to_logical_type();
            let child = if let ArrowDataType::List(inner) = a {
                to_column_write_options_rec(inner, child_overwrites)
            } else if let ArrowDataType::LargeList(inner) = a {
                to_column_write_options_rec(inner, child_overwrites)
            } else if let ArrowDataType::FixedSizeList(inner, _) = a {
                to_column_write_options_rec(inner, child_overwrites)
            } else {
                unreachable!()
            };

            column_options.children =
                ChildWriteOptions::ListLike(Box::new(ListLikeFieldWriteOptions { child }));
        },
        Struct => {
            if let ArrowDataType::Struct(fields) = field.dtype().to_logical_type() {
                let children_overwrites = overwrites.map(|o| match &o.children {
                    ChildFieldOverwrites::Struct(child_overwrites) => PlHashMap::from_iter(
                        child_overwrites
                            .iter()
                            .map(|f| (f.name.as_ref().unwrap(), f)),
                    ),
                    _ => unreachable!(),
                });

                let children = fields
                    .iter()
                    .map(|f| {
                        let overwrites = children_overwrites
                            .as_ref()
                            .and_then(|o| o.get(&f.name).copied());
                        to_column_write_options_rec(f, overwrites)
                    })
                    .collect();

                column_options.children =
                    ChildWriteOptions::Struct(Box::new(StructFieldWriteOptions { children }));
            } else {
                unreachable!()
            }
        },

        Map | Union => unreachable!(),
    }

    column_options
}

pub fn get_column_write_options(
    schema: &ArrowSchema,
    field_overwrites: &[ParquetFieldOverwrites],
) -> Vec<ColumnWriteOptions> {
    let field_overwrites = PlHashMap::from(
        field_overwrites
            .iter()
            .map(|f| (f.name.as_ref().unwrap(), f))
            .collect(),
    );
    schema
        .iter_values()
        .map(|f| to_column_write_options_rec(f, field_overwrites.get(&f.name).copied()))
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
