use std::io::Write;
use std::sync::Arc;

use arrow::datatypes::ArrowSchema;
use polars_error::{PolarsError, PolarsResult};

use super::schema::schema_to_metadata_key;
use super::{ThriftFileMetadata, WriteOptions, to_parquet_schema};
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

    pub fn new_with_parquet_schema_and_encryption(
        writer: W,
        schema: ArrowSchema,
        parquet_schema: SchemaDescriptor,
        options: WriteOptions,
        file_encryption_properties: Arc<FileEncryptionProperties>,
    ) -> PolarsResult<Self> {
        let created_by = Some("Polars".to_string());

        Ok(Self {
            writer: crate::parquet::write::FileWriter::new_with_encryption(
                writer,
                parquet_schema,
                FileWriteOptions {
                    version: options.version,
                    write_statistics: options.has_statistics(),
                },
                created_by,
                file_encryption_properties,
            )?,
            schema,
            options,
        })
    }

    /// Returns a new [`FileWriter`].
    /// # Error
    /// If it is unable to derive a parquet schema from [`ArrowSchema`].
    pub fn try_new(writer: W, schema: ArrowSchema, options: WriteOptions) -> PolarsResult<Self> {
        let parquet_schema = to_parquet_schema(&schema)?;
        Ok(Self::new_with_parquet_schema(
            writer,
            schema,
            parquet_schema,
            options,
        ))
    }

    pub fn try_new_with_encryption(
        writer: W,
        schema: ArrowSchema,
        options: WriteOptions,
        file_encryption_properties: Arc<FileEncryptionProperties>,
    ) -> PolarsResult<Self> {
        let parquet_schema = to_parquet_schema(&schema)?;
        Self::new_with_parquet_schema_and_encryption(
            writer,
            schema,
            parquet_schema,
            options,
            file_encryption_properties,
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
    /// the Arrow schema is added to the metadata.
    pub fn end(&mut self, key_value_metadata: Option<Vec<KeyValue>>) -> PolarsResult<u64> {
        let key_value_metadata =
            key_value_metadata.unwrap_or_else(|| vec![schema_to_metadata_key(&self.schema)]);
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, PrimitiveArray};
    use arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
    use arrow::record_batch::RecordBatchT;
    use polars_buffer::Buffer;
    use polars_utils::pl_str::PlSmallStr;

    use super::super::{
        Encoding, StatisticsOptions, WriteOptions, row_group_iter, to_parquet_schema,
    };
    use super::*;
    use crate::arrow::read::column_iter_to_arrays;
    use crate::parquet::compression::CompressionOptions;
    use crate::parquet::encryption::decrypt::FileDecryptionProperties;
    use crate::parquet::encryption::encrypt::FileEncryptionProperties;
    use crate::parquet::read::{
        BasicDecompressor, PageReader, get_page_iterator, read_metadata_with_size,
        read_metadata_with_size_and_decryption,
    };
    use crate::parquet::write::Version;

    fn test_key() -> Vec<u8> {
        b"0123456789012345".to_vec()
    }

    fn write_options() -> WriteOptions {
        WriteOptions {
            statistics: StatisticsOptions::empty(),
            version: Version::V2,
            compression: CompressionOptions::Uncompressed,
            data_page_size: None,
        }
    }

    fn test_schema() -> ArrowSchema {
        ArrowSchema::from_iter([Field::new(
            PlSmallStr::from("a"),
            ArrowDataType::Int32,
            false,
        )])
    }

    fn test_batch(schema: &ArrowSchema) -> RecordBatchT<ArrayRef> {
        RecordBatchT::new(
            4,
            Arc::new(schema.clone()),
            vec![Box::new(PrimitiveArray::from_vec(vec![1i32, 2, 3, 4])) as ArrayRef],
        )
    }

    fn write_encrypted_file(
        file_encryption_properties: Arc<FileEncryptionProperties>,
        encodings: Buffer<Vec<Encoding>>,
    ) -> Vec<u8> {
        let schema = test_schema();
        let options = write_options();
        let parquet_schema = to_parquet_schema(&schema).unwrap();
        let fields = parquet_schema.fields().to_vec();
        let batch = test_batch(&schema);
        let row_group = row_group_iter(batch, encodings, fields, options);

        let mut writer = FileWriter::try_new_with_encryption(
            Vec::new(),
            schema,
            options,
            file_encryption_properties,
        )
        .unwrap();
        writer.write(4, row_group).unwrap();
        writer.end(None).unwrap();
        writer.into_inner()
    }

    fn read_first_i32_column(bytes: &[u8], decryption_key: Vec<u8>) -> Vec<i32> {
        let decryption_properties = FileDecryptionProperties::builder(decryption_key)
            .build()
            .unwrap();
        let mut metadata_reader = Cursor::new(bytes);
        let metadata = read_metadata_with_size_and_decryption(
            &mut metadata_reader,
            bytes.len() as u64,
            Some(decryption_properties),
        )
        .unwrap();

        let column = &metadata.row_groups[0].parquet_columns()[0];
        let pages = get_page_iterator(
            column,
            Cursor::new(Buffer::from_vec(bytes.to_vec())),
            vec![],
            usize::MAX,
        )
        .unwrap();
        let (arrays, _) = column_iter_to_arrays(
            vec![BasicDecompressor::new(pages, vec![])],
            vec![&column.descriptor().descriptor.primitive_type],
            test_schema().get("a").unwrap().clone(),
            None,
        )
        .unwrap();
        let array = arrays[0]
            .as_any()
            .downcast_ref::<PrimitiveArray<i32>>()
            .unwrap();
        array.values().as_slice().to_vec()
    }

    fn read_first_i32_column_from_chunk(bytes: &[u8], decryption_key: Vec<u8>) -> Vec<i32> {
        let decryption_properties = FileDecryptionProperties::builder(decryption_key)
            .build()
            .unwrap();
        let mut metadata_reader = Cursor::new(bytes);
        let metadata = read_metadata_with_size_and_decryption(
            &mut metadata_reader,
            bytes.len() as u64,
            Some(decryption_properties),
        )
        .unwrap();

        let column = &metadata.row_groups[0].parquet_columns()[0];
        let byte_range = column.byte_range();
        let chunk =
            Buffer::from_vec(bytes[byte_range.start as usize..byte_range.end as usize].to_vec());
        let pages = PageReader::new(Cursor::new(chunk), column, vec![], usize::MAX);
        let (arrays, _) = column_iter_to_arrays(
            vec![BasicDecompressor::new(pages, vec![])],
            vec![&column.descriptor().descriptor.primitive_type],
            test_schema().get("a").unwrap().clone(),
            None,
        )
        .unwrap();
        let array = arrays[0]
            .as_any()
            .downcast_ref::<PrimitiveArray<i32>>()
            .unwrap();
        array.values().as_slice().to_vec()
    }

    #[test]
    fn encrypted_footer_round_trip() {
        let key = test_key();
        let encryption_properties = FileEncryptionProperties::builder(key.clone())
            .build()
            .unwrap();
        let bytes = write_encrypted_file(
            encryption_properties,
            Buffer::from_vec(vec![vec![Encoding::Plain]]),
        );

        assert_eq!(&bytes[..4], b"PARE");
        assert_eq!(&bytes[bytes.len() - 4..], b"PARE");
        assert!(read_metadata_with_size(&mut Cursor::new(&bytes), bytes.len() as u64).is_err());
        assert_eq!(read_first_i32_column(&bytes, key), vec![1, 2, 3, 4]);
    }

    #[test]
    fn encrypted_dictionary_page_round_trip_from_sliced_column_chunk() {
        let key = test_key();
        let encryption_properties = FileEncryptionProperties::builder(key.clone())
            .build()
            .unwrap();
        let bytes = write_encrypted_file(
            encryption_properties,
            Buffer::from_vec(vec![vec![Encoding::RleDictionary]]),
        );

        assert_eq!(
            read_first_i32_column_from_chunk(&bytes, key),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn encrypted_footer_rejects_wrong_key() {
        let encryption_properties = FileEncryptionProperties::builder(test_key())
            .build()
            .unwrap();
        let bytes = write_encrypted_file(
            encryption_properties,
            Buffer::from_vec(vec![vec![Encoding::Plain]]),
        );
        let wrong_key = b"abcdefghijklmnop".to_vec();
        let decryption_properties = FileDecryptionProperties::builder(wrong_key)
            .build()
            .unwrap();

        assert!(
            read_metadata_with_size_and_decryption(
                &mut Cursor::new(&bytes),
                bytes.len() as u64,
                Some(decryption_properties),
            )
            .is_err()
        );
    }

    #[test]
    fn plaintext_footer_encrypted_columns_round_trip() {
        let key = test_key();
        let encryption_properties = FileEncryptionProperties::builder(key.clone())
            .with_plaintext_footer(true)
            .build()
            .unwrap();
        let bytes = write_encrypted_file(
            encryption_properties,
            Buffer::from_vec(vec![vec![Encoding::Plain]]),
        );

        assert_eq!(&bytes[..4], b"PAR1");
        assert_eq!(&bytes[bytes.len() - 4..], b"PAR1");
        read_metadata_with_size(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
        assert_eq!(read_first_i32_column(&bytes, key), vec![1, 2, 3, 4]);
    }

    #[test]
    fn column_key_round_trip() {
        let footer_key = test_key();
        let column_key = b"abcdef0123456789".to_vec();
        let encryption_properties = FileEncryptionProperties::builder(footer_key.clone())
            .with_column_key("a", column_key.clone())
            .build()
            .unwrap();
        let bytes = write_encrypted_file(
            encryption_properties,
            Buffer::from_vec(vec![vec![Encoding::Plain]]),
        );
        let decryption_key = FileDecryptionProperties::builder(footer_key)
            .with_column_key("a", column_key)
            .build()
            .unwrap();

        let mut metadata_reader = Cursor::new(&bytes);
        let metadata = read_metadata_with_size_and_decryption(
            &mut metadata_reader,
            bytes.len() as u64,
            Some(decryption_key),
        )
        .unwrap();
        let column = &metadata.row_groups[0].parquet_columns()[0];
        let pages = get_page_iterator(
            column,
            Cursor::new(Buffer::from_vec(bytes)),
            vec![],
            usize::MAX,
        )
        .unwrap();
        let (arrays, _) = column_iter_to_arrays(
            vec![BasicDecompressor::new(pages, vec![])],
            vec![&column.descriptor().descriptor.primitive_type],
            test_schema().get("a").unwrap().clone(),
            None,
        )
        .unwrap();
        let array = arrays[0]
            .as_any()
            .downcast_ref::<PrimitiveArray<i32>>()
            .unwrap();
        assert_eq!(array.values().as_slice(), &[1, 2, 3, 4]);
    }
}
