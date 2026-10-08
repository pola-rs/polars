//! Tests for writing encrypted files, checking parts of the file that aren't
//! exercised by reading files back, such as the page indexes.

use std::io::Cursor;
use std::sync::Arc;

use polars_arrow::array::{Array, Int64Array};
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
use polars_arrow::record_batch::RecordBatchT;
use polars_buffer::Buffer;
use polars_parquet_format::thrift::protocol::{ReadThrift, TCompactInputProtocol};
use polars_parquet_format::{ColumnIndex, OffsetIndex};

use super::ciphers::{AesGcmBlockDecryptor, BlockDecryptor};
use super::decrypt::{FileDecryptionProperties, decrypt_module};
use super::encrypt::FileEncryptionProperties;
use super::modules::{ModuleType, create_module_aad};
use crate::arrow::write::{
    CompressionOptions, Encoding, FileWriter, RowGroupIterator, StatisticsOptions, Version,
    WriteOptions,
};
use crate::parquet::metadata::{ColumnChunkMetadata, FileMetadata};
use crate::parquet::read::read_metadata_with_decryption;
use crate::parquet::{ENCRYPTED_PARQUET_MAGIC, PARQUET_MAGIC};

const FOOTER_KEY: &[u8] = b"0123456789012345";
const COLUMN_KEY: &[u8] = b"1234567890123450";
const NUM_ROW_GROUPS: usize = 2;

/// Write a file with two Int64 columns, `a` and `b`, and multiple row groups.
fn write_file(encryption_properties: Arc<FileEncryptionProperties>) -> Vec<u8> {
    let schema = ArrowSchema::from_iter(
        ["a", "b"].map(|name| Field::new(name.into(), ArrowDataType::Int64, true)),
    );
    let columns: Vec<Box<dyn Array>> = vec![
        Int64Array::from_iter((0..100).map(Some)).boxed(),
        Int64Array::from_iter((100..200).map(Some)).boxed(),
    ];
    let options = WriteOptions {
        statistics: StatisticsOptions::full(),
        compression: CompressionOptions::Uncompressed,
        version: Version::V1,
        data_page_size: None,
    };
    let batches = (0..NUM_ROW_GROUPS)
        .map(|_| RecordBatchT::try_new(100, Arc::new(schema.clone()), columns.clone()));
    let row_groups = RowGroupIterator::try_new(
        batches,
        &schema,
        options,
        Buffer::from_iter([vec![Encoding::Plain], vec![Encoding::Plain]]),
    )
    .unwrap();

    let mut writer = FileWriter::try_new(Cursor::new(vec![]), schema.clone(), options)
        .unwrap()
        .with_encryption_properties(Some(encryption_properties))
        .unwrap();
    for row_group in row_groups {
        writer.write(100, row_group.unwrap()).unwrap();
    }
    writer.end(None).unwrap();
    writer.into_inner().into_inner()
}

fn read_metadata(
    data: &[u8],
    decryption_properties: &Arc<FileDecryptionProperties>,
) -> FileMetadata {
    read_metadata_with_decryption(&mut Cursor::new(data), Some(decryption_properties), None)
        .unwrap()
}

/// Read and decrypt a module, which is a column or offset index.
fn decrypt_index_module(
    data: &[u8],
    offset: Option<i64>,
    length: Option<i32>,
    key: &[u8],
    aad: &[u8],
) -> Vec<u8> {
    let offset = offset.unwrap() as usize;
    let module = &data[offset..offset + length.unwrap() as usize];
    let decryptor: Arc<dyn BlockDecryptor> = Arc::new(AesGcmBlockDecryptor::new(key).unwrap());
    let (plaintext, consumed) = decrypt_module(&decryptor, module, aad).unwrap();
    assert_eq!(consumed, module.len());
    plaintext
}

fn read_index_module(data: &[u8], offset: Option<i64>, length: Option<i32>) -> Vec<u8> {
    let offset = offset.unwrap() as usize;
    data[offset..offset + length.unwrap() as usize].to_vec()
}

fn decode<T: ReadThrift>(bytes: &[u8]) -> T {
    T::read_from_in_protocol(&mut TCompactInputProtocol::new(bytes, usize::MAX)).unwrap()
}

#[test]
fn test_file_magic() {
    let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
        .build()
        .unwrap();
    let data = write_file(properties);
    assert_eq!(data[..4], ENCRYPTED_PARQUET_MAGIC);
    assert_eq!(data[data.len() - 4..], ENCRYPTED_PARQUET_MAGIC);

    let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
        .with_plaintext_footer(true)
        .build()
        .unwrap();
    let data = write_file(properties);
    assert_eq!(data[..4], PARQUET_MAGIC);
    assert_eq!(data[data.len() - 4..], PARQUET_MAGIC);
}

#[test]
fn test_page_indexes_encrypted() {
    // Only column `a` is encrypted
    let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
        .with_column_key("a", COLUMN_KEY.to_vec())
        .build()
        .unwrap();
    let data = write_file(properties);
    let decryption_properties = FileDecryptionProperties::builder(FOOTER_KEY.to_vec())
        .with_column_key("a", COLUMN_KEY.to_vec())
        .build()
        .unwrap();
    let metadata = read_metadata(&data, &decryption_properties);
    let file_aad = metadata.decryptor.as_ref().unwrap().file_aad().clone();
    assert_eq!(metadata.row_groups.len(), NUM_ROW_GROUPS);

    for (row_group_idx, row_group) in metadata.row_groups.iter().enumerate() {
        let columns: &[ColumnChunkMetadata] = row_group.parquet_columns();
        let (a, b) = (&columns[0], &columns[1]);
        assert!(a.is_encrypted());
        assert!(!b.is_encrypted());

        let aad = |module_type| {
            create_module_aad(&file_aad, module_type, row_group_idx, 0, None).unwrap()
        };
        let column_index: ColumnIndex = decode(&decrypt_index_module(
            &data,
            a.column_index_offset(),
            a.column_index_length(),
            COLUMN_KEY,
            &aad(ModuleType::ColumnIndex),
        ));
        assert!(!column_index.min_values.is_empty());
        let offset_index: OffsetIndex = decode(&decrypt_index_module(
            &data,
            a.offset_index_offset(),
            a.offset_index_length(),
            COLUMN_KEY,
            &aad(ModuleType::OffsetIndex),
        ));
        assert!(!offset_index.page_locations.is_empty());

        // Indexes for the unencrypted column are not encrypted
        let column_index: ColumnIndex = decode(&read_index_module(
            &data,
            b.column_index_offset(),
            b.column_index_length(),
        ));
        assert!(!column_index.min_values.is_empty());
        let offset_index: OffsetIndex = decode(&read_index_module(
            &data,
            b.offset_index_offset(),
            b.offset_index_length(),
        ));
        assert!(!offset_index.page_locations.is_empty());
    }
}

#[test]
fn test_uniform_encryption_page_indexes_use_footer_key() {
    let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
        .build()
        .unwrap();
    let data = write_file(properties);
    let decryption_properties = FileDecryptionProperties::builder(FOOTER_KEY.to_vec())
        .build()
        .unwrap();
    let metadata = read_metadata(&data, &decryption_properties);
    let file_aad = metadata.decryptor.as_ref().unwrap().file_aad().clone();

    for (row_group_idx, row_group) in metadata.row_groups.iter().enumerate() {
        for (column_ordinal, column) in row_group.parquet_columns().iter().enumerate() {
            assert!(column.is_encrypted());
            let aad = create_module_aad(
                &file_aad,
                ModuleType::ColumnIndex,
                row_group_idx,
                column_ordinal,
                None,
            )
            .unwrap();
            let column_index: ColumnIndex = decode(&decrypt_index_module(
                &data,
                column.column_index_offset(),
                column.column_index_length(),
                FOOTER_KEY,
                &aad,
            ));
            assert!(!column_index.min_values.is_empty());
        }
    }
}
