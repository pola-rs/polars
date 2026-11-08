use std::io::Write;
use std::sync::Arc;

use polars_arrow::datatypes::ArrowSchema;
use polars_error::{PolarsError, PolarsResult};
use polars_utils::version::{
    get_polars_lib_build_commit, get_polars_lib_name, get_polars_lib_version,
};

use super::schema::schema_to_metadata_key;
use super::{ThriftFileMetadata, WriteOptions, to_parquet_schema};
use crate::arrow::geo::geo_key_value;
use crate::parquet::encryption::encrypt::FileEncryptionProperties;
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

    /// The metadata of the written file. None until the file has ended, and never set for encrypted files.
    pub fn metadata(&self) -> Option<&ThriftFileMetadata> {
        self.writer.metadata()
    }
}

impl<W: Write> FileWriter<W> {
    /// Returns a new [`FileWriter`].
    ///
    /// # Errors
    /// If encryption properties are set and are not valid for the file schema.
    pub fn try_new_with_parquet_schema(
        writer: W,
        schema: ArrowSchema,
        parquet_schema: SchemaDescriptor,
        options: WriteOptions,
        encryption_properties: Option<Arc<FileEncryptionProperties>>,
    ) -> PolarsResult<Self> {
        let created_by = Some(format!(
            "{} version {} (build {})",
            get_polars_lib_name(),
            get_polars_lib_version(),
            get_polars_lib_build_commit()
        ));

        Ok(Self {
            writer: crate::parquet::write::FileWriter::try_new(
                writer,
                parquet_schema,
                FileWriteOptions {
                    version: options.version,
                    write_statistics: options.has_statistics(),
                },
                created_by,
                encryption_properties,
            )?,
            schema,
            options,
        })
    }

    /// Returns a new [`FileWriter`].
    ///
    /// # Error
    /// If it is unable to derive a parquet schema from [`ArrowSchema`], or the encryption
    /// properties are set and not valid for the file schema.
    pub fn try_new(
        writer: W,
        schema: ArrowSchema,
        options: WriteOptions,
        encryption_properties: Option<Arc<FileEncryptionProperties>>,
    ) -> PolarsResult<Self> {
        let parquet_schema = to_parquet_schema(&schema)?;
        Self::try_new_with_parquet_schema(
            writer,
            schema,
            parquet_schema,
            options,
            encryption_properties,
        )
    }

    /// Writes a row group to the file.
    pub fn write(
        &mut self,
        num_rows: u64,
        row_group: RowGroupIterColumns<'_, PolarsError>,
    ) -> PolarsResult<()> {
        Ok(self.writer.write(num_rows, row_group)?)
    }

    /// Writes the footer of the parquet file. Returns the total size of the file.
    /// If `key_value_metadata` is provided, the value is taken as-is. If it is not provided,
    /// the Arrow schema is added to the metadata. GeoParquet `geo` metadata is added if enabled
    /// and not already present.
    pub fn end(&mut self, key_value_metadata: Option<Vec<KeyValue>>) -> PolarsResult<u64> {
        let mut key_value_metadata =
            key_value_metadata.unwrap_or_else(|| vec![schema_to_metadata_key(&self.schema)]);
        if let Some(geo) = geo_key_value(&self.schema)?
            && !key_value_metadata.iter().any(|kv| kv.key == geo.key)
        {
            key_value_metadata.push(geo);
        }
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
