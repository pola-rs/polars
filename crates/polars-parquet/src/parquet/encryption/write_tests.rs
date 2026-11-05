//! Tests for writing encrypted files, checking parts of the file that aren't
//! exercised by reading files back, such as the page indexes, and reading keys
//! from key metadata with a [`KeyRetriever`].

use std::io::Cursor;
use std::sync::{Arc, Mutex};

use polars_arrow::array::{Array, Int64Array};
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
use polars_arrow::record_batch::RecordBatchT;
use polars_buffer::Buffer;
use polars_parquet_format::thrift::protocol::{ReadThrift, TCompactInputProtocol};
use polars_parquet_format::{ColumnIndex, OffsetIndex};
use polars_utils::aliases::PlHashMap;

use super::ciphers::{AesGcmBlockDecryptor, BlockDecryptor};
use super::decrypt::{FileDecryptionProperties, KeyRetriever, decrypt_module};
use super::encrypt::FileEncryptionProperties;
use super::modules::{ModuleType, create_module_aad};
use crate::arrow::write::{
    CompressionOptions, Encoding, FileWriter, RowGroupIterator, StatisticsOptions, Version,
    WriteOptions, to_parquet_schema,
};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnChunkMetadata, FileMetadata};
use crate::parquet::page::CompressedPage;
use crate::parquet::read::{get_page_iterator, read_metadata_with_decryption};
use crate::parquet::{ENCRYPTED_PARQUET_MAGIC, PARQUET_MAGIC};

const FOOTER_KEY: &[u8] = b"0123456789012345";
const COLUMN_KEY: &[u8] = b"1234567890123450";
const FOOTER_KEY_METADATA: &[u8] = b"footer_key";
const COLUMN_KEY_METADATA: &[u8] = b"column_key";
const NUM_ROW_GROUPS: usize = 2;

/// Write a file with two Int64 columns, `a` and `b`, and multiple row groups.
fn write_file(encryption_properties: Arc<FileEncryptionProperties>) -> Vec<u8> {
    write_file_with_writer(Some(encryption_properties))
        .into_inner()
        .into_inner()
}

/// Write the file from [`write_file`], returning the ended writer.
fn write_file_with_writer(
    encryption_properties: Option<Arc<FileEncryptionProperties>>,
) -> FileWriter<Cursor<Vec<u8>>> {
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

    let mut writer = FileWriter::try_new(
        Cursor::new(vec![]),
        schema.clone(),
        options,
        encryption_properties,
    )
    .unwrap();
    for row_group in row_groups {
        writer.write(100, row_group.unwrap()).unwrap();
    }
    writer.end(None).unwrap();
    writer
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

/// Read the values of an Int64 column chunk written by [`write_file`].
fn read_column_values(data: &[u8], column: &ColumnChunkMetadata) -> ParquetResult<Vec<i64>> {
    let reader = Cursor::new(Buffer::from(data.to_vec()));
    let pages = get_page_iterator(column, reader, vec![], usize::MAX)?
        .collect::<ParquetResult<Vec<_>>>()?;
    let [CompressedPage::Data(page)] = pages.as_slice() else {
        panic!("expected a single data page");
    };
    // The page is uncompressed and plain encoded, with the values after the
    // definition levels.
    let values = &page.buffer[page.buffer.len() - 8 * page.num_values()..];
    Ok(values
        .chunks_exact(8)
        .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

/// Look up keys by their key metadata, recording the metadata of each request.
struct TestKeyRetriever {
    keys: PlHashMap<Vec<u8>, Vec<u8>>,
    requests: Mutex<Vec<Vec<u8>>>,
}

impl TestKeyRetriever {
    fn new(keys: &[(&[u8], &[u8])]) -> Arc<Self> {
        Arc::new(Self {
            keys: keys
                .iter()
                .map(|(metadata, key)| (metadata.to_vec(), key.to_vec()))
                .collect(),
            requests: Mutex::new(vec![]),
        })
    }

    fn requested(&self, key_metadata: &[u8]) -> bool {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .any(|m| m == key_metadata)
    }
}

impl KeyRetriever for TestKeyRetriever {
    fn retrieve_key(&self, key_metadata: &[u8]) -> ParquetResult<Vec<u8>> {
        self.requests.lock().unwrap().push(key_metadata.to_vec());
        self.keys.get(key_metadata).cloned().ok_or_else(|| {
            ParquetError::Encryption(format!(
                "no key for metadata {:?}",
                String::from_utf8_lossy(key_metadata)
            ))
        })
    }
}

/// Write a file where the footer key and the key for column `a` have key metadata.
fn write_file_with_key_metadata(plaintext_footer: bool) -> Vec<u8> {
    let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
        .with_footer_key_metadata(FOOTER_KEY_METADATA.to_vec())
        .with_column_key_and_metadata("a", COLUMN_KEY.to_vec(), COLUMN_KEY_METADATA.to_vec())
        .with_plaintext_footer(plaintext_footer)
        .build()
        .unwrap();
    write_file(properties)
}

fn read_with_retriever(
    data: &[u8],
    retriever: Arc<TestKeyRetriever>,
) -> ParquetResult<FileMetadata> {
    let properties = FileDecryptionProperties::with_key_retriever(retriever)
        .build()
        .unwrap();
    read_metadata_with_decryption(&mut Cursor::new(data), Some(&properties), None)
}

/// Check that both columns of every row group can be read.
fn check_all_values(data: &[u8], metadata: &FileMetadata) {
    for row_group in &metadata.row_groups {
        let columns = row_group.parquet_columns();
        assert!(columns[0].is_encrypted());
        assert!(!columns[1].is_encrypted());
        assert_eq!(
            read_column_values(data, &columns[0]).unwrap(),
            (0..100).collect::<Vec<_>>()
        );
        assert_eq!(
            read_column_values(data, &columns[1]).unwrap(),
            (100..200).collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_no_plaintext_metadata_for_encrypted_file() {
    assert!(write_file_with_writer(None).metadata().is_some());

    for plaintext_footer in [false, true] {
        let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
            .with_plaintext_footer(plaintext_footer)
            .build()
            .unwrap();
        assert!(
            write_file_with_writer(Some(properties))
                .metadata()
                .is_none()
        );
    }
}

/// Validate column keys for `columns` against `schema`, returning any error message.
fn validate_column_names(schema: &ArrowSchema, columns: &[&str]) -> Result<(), String> {
    let mut builder = FileEncryptionProperties::builder(FOOTER_KEY.to_vec());
    for column in columns {
        builder = builder.with_column_key(column, COLUMN_KEY.to_vec());
    }
    builder
        .build()
        .unwrap()
        .validate_encrypted_column_names(&to_parquet_schema(schema).unwrap())
        .map_err(|e| e.to_string())
}

#[test]
fn test_validate_encrypted_column_names() {
    let schema = ArrowSchema::from_iter([
        Field::new("a".into(), ArrowDataType::Int64, true),
        Field::new(
            "s".into(),
            ArrowDataType::Struct(vec![Field::new("x".into(), ArrowDataType::Int64, true)]),
            true,
        ),
        Field::new(
            "l".into(),
            ArrowDataType::LargeList(Box::new(Field::new(
                "item".into(),
                ArrowDataType::Int64,
                true,
            ))),
            true,
        ),
    ]);
    let validate = |columns: &[&str]| validate_column_names(&schema, columns);

    assert_eq!(validate(&[]), Ok(()));
    assert_eq!(validate(&["a", "s.x", "l.list.element"]), Ok(()));
    // Only leaf column paths match, and matching is case sensitive. Missing columns
    // are listed in sorted order.
    for (columns, missing) in [
        (&["a", "missing"][..], "missing"),
        (&["s"], "s"),
        (&["l"], "l"),
        (&["l.element"], "l.element"),
        (&["A"], "A"),
        (&["z", "a", "s"], "s, z"),
    ] {
        let err = validate(columns).unwrap_err();
        assert!(
            err.ends_with(&format!(
                "columns with encryption keys specified were not found in the schema: {missing}"
            )),
            "{err}"
        );
    }

    // The writer validates the columns
    let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
        .with_column_key("missing", COLUMN_KEY.to_vec())
        .build()
        .unwrap();
    assert!(
        FileWriter::try_new(
            Cursor::new(vec![]),
            schema,
            WriteOptions {
                statistics: StatisticsOptions::full(),
                compression: CompressionOptions::Uncompressed,
                version: Version::V1,
                data_page_size: None,
            },
            Some(properties),
        )
        .is_err()
    );
}

#[test]
fn test_validate_ambiguous_column_names() {
    // A top-level column named `a.b` and field `b` of struct `a` have the same path
    let schema = ArrowSchema::from_iter([
        Field::new("a.b".into(), ArrowDataType::Int64, true),
        Field::new(
            "a".into(),
            ArrowDataType::Struct(vec![
                Field::new("b".into(), ArrowDataType::Int64, true),
                Field::new("c".into(), ArrowDataType::Int64, true),
            ]),
            true,
        ),
        Field::new("d".into(), ArrowDataType::Int64, true),
    ]);
    let ambiguous_err = "columns with encryption keys specified match multiple columns with the \
        same dot-separated path, so can't be encrypted separately: a.b";

    let err = validate_column_names(&schema, &["a.b"]).unwrap_err();
    assert!(err.ends_with(ambiguous_err), "{err}");
    let err = validate_column_names(&schema, &["a.b", "d"]).unwrap_err();
    assert!(err.ends_with(ambiguous_err), "{err}");
    // Missing columns are reported first
    let err = validate_column_names(&schema, &["a.b", "missing"]).unwrap_err();
    assert!(err.ends_with("not found in the schema: missing"), "{err}");
    // Columns without keys may have the same path
    assert_eq!(validate_column_names(&schema, &[]), Ok(()));
    assert_eq!(validate_column_names(&schema, &["a.c", "d"]), Ok(()));
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

#[test]
fn test_key_retriever_encrypted_footer() {
    let data = write_file_with_key_metadata(false);
    assert_eq!(data[..4], ENCRYPTED_PARQUET_MAGIC);

    // The footer key comes from the key metadata in `FileCryptoMetaData`.
    let retriever = TestKeyRetriever::new(&[
        (FOOTER_KEY_METADATA, FOOTER_KEY),
        (COLUMN_KEY_METADATA, COLUMN_KEY),
    ]);
    let metadata = read_with_retriever(&data, retriever.clone()).unwrap();
    assert!(retriever.requested(FOOTER_KEY_METADATA));
    check_all_values(&data, &metadata);
    assert!(retriever.requested(COLUMN_KEY_METADATA));

    // The wrong footer key can't decrypt the footer.
    let retriever = TestKeyRetriever::new(&[(FOOTER_KEY_METADATA, COLUMN_KEY)]);
    let err = read_with_retriever(&data, retriever).unwrap_err();
    assert!(
        err.to_string().contains("unable to decrypt parquet footer"),
        "{err}"
    );
}

#[test]
fn test_key_retriever_plaintext_footer() {
    let data = write_file_with_key_metadata(true);
    assert_eq!(data[..4], PARQUET_MAGIC);

    // The footer key comes from `footer_signing_key_metadata`, and is used to
    // verify the footer signature.
    let retriever = TestKeyRetriever::new(&[
        (FOOTER_KEY_METADATA, FOOTER_KEY),
        (COLUMN_KEY_METADATA, COLUMN_KEY),
    ]);
    let metadata = read_with_retriever(&data, retriever.clone()).unwrap();
    assert!(retriever.requested(FOOTER_KEY_METADATA));
    check_all_values(&data, &metadata);
    assert!(retriever.requested(COLUMN_KEY_METADATA));

    // The wrong footer key fails signature verification.
    let retriever = TestKeyRetriever::new(&[
        (FOOTER_KEY_METADATA, COLUMN_KEY),
        (COLUMN_KEY_METADATA, COLUMN_KEY),
    ]);
    let err = read_with_retriever(&data, retriever).unwrap_err();
    assert!(
        err.to_string()
            .contains("Footer signature verification failed"),
        "{err}"
    );
}

#[test]
fn test_key_retriever_missing_column_key() {
    for plaintext_footer in [false, true] {
        let data = write_file_with_key_metadata(plaintext_footer);
        // The retriever refuses the column key.
        let retriever = TestKeyRetriever::new(&[(FOOTER_KEY_METADATA, FOOTER_KEY)]);
        let metadata = read_with_retriever(&data, retriever.clone()).unwrap();

        for row_group in &metadata.row_groups {
            let columns = row_group.parquet_columns();
            // The unencrypted column can still be read.
            assert_eq!(
                read_column_values(&data, &columns[1]).unwrap(),
                (100..200).collect::<Vec<_>>()
            );
            let err = read_column_values(&data, &columns[0]).unwrap_err();
            assert!(
                err.to_string()
                    .contains("no key for metadata \"column_key\""),
                "{err}"
            );
        }
        assert!(retriever.requested(COLUMN_KEY_METADATA));
    }
}
