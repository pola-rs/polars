use std::io::Write;

use arrow::datatypes::ArrowSchema;
use polars_error::{PolarsError, PolarsResult};

use super::schema::schema_to_metadata_key;
use super::{ColumnWriteOptions, ThriftFileMetadata, WriteOptions, to_parquet_schema};
use crate::parquet::metadata::{KeyValue, SchemaDescriptor};
use crate::parquet::write::{RowGroupIterColumns, WriteOptions as FileWriteOptions};

/// An interface to write a parquet to a [`Write`]
pub struct FileWriter<W: Write> {
    writer: crate::parquet::write::FileWriter<W>,
    schema: ArrowSchema,
    options: WriteOptions,
}

// Accessors
impl<W: Write> FileWriter<W> {
    /// The options assigned to the file
    pub fn options(&self) -> WriteOptions {
        self.options
    }

    /// The [`SchemaDescriptor`] assigned to this file
    pub fn parquet_schema(&self) -> &SchemaDescriptor {
        self.writer.schema()
    }

    /// The [`ArrowSchema`] assigned to this file
    pub fn schema(&self) -> &ArrowSchema {
        &self.schema
    }
}

impl<W: Write> FileWriter<W> {
    /// Returns a new [`FileWriter`].
    /// # Error
    /// If it is unable to derive a parquet schema from [`ArrowSchema`].
    pub fn new_with_parquet_schema(
        writer: W,
        schema: ArrowSchema,
        parquet_schema: SchemaDescriptor,
        options: WriteOptions,
    ) -> Self {
        let created_by = Some("Polars".to_string());

        Self {
            writer: crate::parquet::write::FileWriter::new(
                writer,
                parquet_schema,
                FileWriteOptions {
                    version: options.version,
                    write_statistics: options.has_statistics(),
                },
                created_by,
            ),
            schema,
            options,
        }
    }

    /// Returns a new [`FileWriter`].
    /// # Error
    /// If it is unable to derive a parquet schema from [`ArrowSchema`].
    pub fn try_new(
        writer: W,
        schema: ArrowSchema,
        options: WriteOptions,
        column_options: &[ColumnWriteOptions],
    ) -> PolarsResult<Self> {
        let parquet_schema = to_parquet_schema(&schema, column_options)?;
        Ok(Self::new_with_parquet_schema(
            writer,
            schema,
            parquet_schema,
            options,
        ))
    }

    /// Writes a row group to the file.
    pub fn write(&mut self, row_group: RowGroupIterColumns<'_, PolarsError>) -> PolarsResult<()> {
        Ok(self.writer.write(row_group)?)
    }

    /// Writes the footer of the parquet file. Returns the total size of the file.
    /// If `key_value_metadata` is provided, the value is taken as-is. If it is not provided,
    /// the Arrow schema is added to the metadata.
    pub fn end(
        &mut self,
        key_value_metadata: Option<Vec<KeyValue>>,
        column_options: &[ColumnWriteOptions],
    ) -> PolarsResult<u64> {
        let key_value_metadata = key_value_metadata
            .unwrap_or_else(|| vec![schema_to_metadata_key(&self.schema, column_options)]);
        Ok(self.writer.end(Some(key_value_metadata))?)
    }

    /// Consumes this writer and returns the inner writer
    pub fn into_inner(self) -> W {
        self.writer.into_inner()
    }

    /// Returns the underlying writer and [`ThriftFileMetadata`]
    /// # Panics
    /// This function panics if [`Self::end`] has not yet been called
    pub fn into_inner_and_metadata(self) -> (W, ThriftFileMetadata) {
        self.writer.into_inner_and_metadata()
    }
}
