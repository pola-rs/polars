use std::io::Write;

use parquet2::metadata::KeyValue;
use parquet2::metadata::SchemaDescriptor;
use parquet2::write::RowGroupIter;
use parquet2::write::WriteOptions as FileWriteOptions;

use crate::datatypes::Schema;
use crate::error::{Error, Result};

use super::{schema::schema_to_metadata_key, to_parquet_schema, ThriftFileMetaData, WriteOptions};

/// Attaches [`Schema`] to `key_value_metadata`
pub fn add_arrow_schema(
    schema: &Schema,
    key_value_metadata: Option<Vec<KeyValue>>,
) -> Option<Vec<KeyValue>> {
    key_value_metadata
        .map(|mut x| {
            x.push(schema_to_metadata_key(schema));
            x
        })
        .or_else(|| Some(vec![schema_to_metadata_key(schema)]))
}

/// An interface to write a parquet to a [`Write`]
pub struct FileWriter<W: Write> {
    writer: parquet2::write::FileWriter<W>,
    schema: Schema,
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

    /// The [`Schema`] assigned to this file
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

impl<W: Write> FileWriter<W> {
    /// Returns a new [`FileWriter`].
    /// # Error
    /// If it is unable to derive a parquet schema from [`Schema`].
    pub fn try_new(writer: W, schema: Schema, options: WriteOptions) -> Result<Self> {
        let parquet_schema = to_parquet_schema(&schema)?;

        let created_by = Some("Arrow2 - Native Rust implementation of Arrow".to_string());

        Ok(Self {
            writer: parquet2::write::FileWriter::new(
                writer,
                parquet_schema,
                FileWriteOptions {
                    version: options.version,
                    write_statistics: options.write_statistics,
                },
                created_by,
            ),
            schema,
            options,
        })
    }

    /// Writes a row group to the file.
    pub fn write(&mut self, row_group: RowGroupIter<'_, Error>) -> Result<()> {
        Ok(self.writer.write(row_group)?)
    }

    /// Writes the footer of the parquet file. Returns the total size of the file.
    pub fn end(&mut self, key_value_metadata: Option<Vec<KeyValue>>) -> Result<u64> {
        let key_value_metadata = add_arrow_schema(&self.schema, key_value_metadata);
        Ok(self.writer.end(key_value_metadata)?)
    }

    /// Consumes this writer and returns the inner writer
    pub fn into_inner(self) -> W {
        self.writer.into_inner()
    }

    /// Returns the underlying writer and [`ThriftFileMetaData`]
    /// # Panics
    /// This function panics if [`Self::end`] has not yet been called
    pub fn into_inner_and_metadata(self) -> (W, ThriftFileMetaData) {
        self.writer.into_inner_and_metadata()
    }
}
