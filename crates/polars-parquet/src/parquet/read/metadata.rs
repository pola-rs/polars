use std::cmp::min;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_parquet_format::FileCryptoMetaData;

use super::super::metadata::FileMetadata;
use super::super::{
    DEFAULT_FOOTER_READ_SIZE, ENCRYPTED_PARQUET_MAGIC, FOOTER_SIZE, HEADER_SIZE, PARQUET_MAGIC,
};
use crate::parquet::encryption::decrypt::{FileDecryptionProperties, FileDecryptor};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::handwritten_thrift::{
    decode_file_crypto_metadata, decode_file_metadata, decode_num_rows,
};

pub(super) fn metadata_len(buffer: &[u8]) -> u32 {
    let len = buffer.len();
    u32::from_le_bytes(buffer[len - 8..len - 4].try_into().unwrap())
}

// see (unstable) Seek::stream_len
fn stream_len(seek: &mut impl Seek) -> std::result::Result<u64, std::io::Error> {
    let old_pos = seek.stream_position()?;
    let len = seek.seek(SeekFrom::End(0))?;

    // Avoid seeking a third time when we were already at the end of the
    // stream. The branch is usually way cheaper than a seek operation.
    if old_pos != len {
        seek.seek(SeekFrom::Start(old_pos))?;
    }

    Ok(len)
}

/// Reads a [`FileMetadata`] from the reader, located at the end of the file.
pub fn read_metadata<R: Read + Seek>(reader: &mut R) -> ParquetResult<FileMetadata> {
    // check file is large enough to hold footer
    let file_size = stream_len(reader)?;
    read_metadata_with_size(reader, file_size)
}

/// Reads a [`FileMetadata`] from the reader, located at the end of the file, with known file size.
pub fn read_metadata_with_size<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
) -> ParquetResult<FileMetadata> {
    let footer = fetch_footer_buf(reader, file_size)?;
    deserialize_metadata(footer.into_plaintext()?)
}

/// Reads a [`FileMetadata`] from the reader, located at the end of the file,
/// using the provided decryption properties if the file is encrypted.
pub fn read_metadata_with_decryption<R: Read + Seek>(
    reader: &mut R,
    decryption_properties: Option<&Arc<FileDecryptionProperties>>,
    file_size: Option<u64>,
) -> ParquetResult<FileMetadata> {
    let file_size = match file_size {
        Some(file_size) => file_size,
        None => stream_len(reader)?,
    };
    let footer = fetch_footer_buf(reader, file_size)?;
    decode_footer(footer, decryption_properties)
}

/// Parse loaded metadata bytes via the hand-written Thrift compact decoder.
///
/// `footer` must be a [`Buffer<u8>`] because [`FileMetadata`] holds the buffer
/// for the lifetime of the metadata; column-chunk statistics store
/// `ByteRange`s into it instead of allocating per-stat byte vecs.
pub fn deserialize_metadata(footer: Buffer<u8>) -> ParquetResult<FileMetadata> {
    let compact = decode_file_metadata(footer)?;
    FileMetadata::from_compact(compact)
}

/// Parse loaded metadata bytes, using the provided decryption properties if the file
/// is encrypted.
///
/// `footer` must include the trailing metadata length and magic bytes, which are used
/// to determine whether the footer is encrypted.
///
/// Files with an encrypted footer require the decryption properties to read the metadata.
/// Files with a plaintext footer may still have encrypted columns. If decryption properties
/// are provided, the footer signature is verified unless disabled in the decryption properties.
pub fn deserialize_metadata_with_decryption(
    footer: Buffer<u8>,
    decryption_properties: Option<&Arc<FileDecryptionProperties>>,
) -> ParquetResult<FileMetadata> {
    decode_footer(FooterBuffer::try_new(footer)?, decryption_properties)
}

fn decode_footer(
    footer: FooterBuffer,
    decryption_properties: Option<&Arc<FileDecryptionProperties>>,
) -> ParquetResult<FileMetadata> {
    let Some(decryption_properties) = decryption_properties else {
        return deserialize_metadata(footer.into_plaintext()?);
    };

    if footer.encrypted {
        // First read the FileCryptoMetaData, which comes before the encrypted footer and is
        // needed to decrypt the footer.
        let (crypto_metadata, encrypted_footer) =
            deserialize_file_crypto_metadata(footer.metadata())?;
        let decryptor = FileDecryptor::from_encryption_algorithm(
            decryption_properties,
            crypto_metadata.encryption_algorithm,
            crypto_metadata.key_metadata.as_deref(),
        )?;
        let footer = Buffer::from_vec(decryptor.decrypt_footer(&encrypted_footer)?);
        let compact = decode_file_metadata(footer)?;
        FileMetadata::from_compact_with_decryptor(compact, Some(Arc::new(decryptor)))
    } else {
        let mut compact = decode_file_metadata(footer.buffer.clone())?;
        // A file with a plaintext footer may have encrypted columns and require a file decryptor.
        // In this case the FileMetaData stores the EncryptionAlgorithm instead of FileCryptoMetaData.
        let decryptor = compact
            .encryption_algorithm
            .take()
            .map(|algorithm| {
                let decryptor = FileDecryptor::from_encryption_algorithm(
                    decryption_properties,
                    algorithm,
                    compact.footer_signing_key_metadata.as_deref(),
                )?;
                if decryption_properties.check_plaintext_footer_integrity() {
                    decryptor.verify_plaintext_footer_signature(&footer.metadata())?;
                }
                Ok::<_, ParquetError>(Arc::new(decryptor))
            })
            .transpose()?;
        FileMetadata::from_compact_with_decryptor(compact, decryptor)
    }
}

/// Parses the file crypto metadata of a Parquet file with an encrypted footer,
/// and returns the remaining encrypted footer bytes.
///
/// `footer` must exclude the trailing metadata length and magic bytes.
pub fn deserialize_file_crypto_metadata(
    footer: Buffer<u8>,
) -> ParquetResult<(FileCryptoMetaData, Buffer<u8>)> {
    let (crypto_metadata, consumed) = decode_file_crypto_metadata(&footer)?;
    Ok((crypto_metadata, footer.sliced(consumed..)))
}

/// Decode only `FileMetaData.num_rows` (thrift field 3) from `footer`.
/// Used by Polars multi-file scans in `RowCounts` resolve mode. See
/// [`crate::parquet::handwritten_thrift::decode_num_rows`].
pub fn deserialize_num_rows(footer: Buffer<u8>) -> ParquetResult<i64> {
    decode_num_rows(footer)
}

/// Sync variant of [`deserialize_num_rows`] that owns the reader.
pub fn read_num_rows<R: Read + Seek>(reader: &mut R) -> ParquetResult<i64> {
    let file_size = stream_len(reader)?;
    read_num_rows_with_size(reader, file_size)
}

/// As [`read_num_rows`] but with a pre-fetched file size.
pub(crate) fn read_num_rows_with_size<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
) -> ParquetResult<i64> {
    let footer = fetch_footer_buf(reader, file_size)?;
    decode_num_rows(footer.into_plaintext()?)
}

struct FooterBuffer {
    /// The footer bytes, including the trailing metadata length and magic bytes
    buffer: Buffer<u8>,
    /// Whether the footer is encrypted
    encrypted: bool,
}

impl FooterBuffer {
    /// Create from footer bytes that include the trailing metadata length and magic bytes.
    fn try_new(buffer: Buffer<u8>) -> ParquetResult<Self> {
        if buffer.len() < FOOTER_SIZE as usize {
            return Err(ParquetError::oos(format!(
                "The footer must be at least {FOOTER_SIZE} bytes"
            )));
        }
        let encrypted = is_encrypted_footer(&buffer[buffer.len() - PARQUET_MAGIC.len()..])?;
        Ok(Self { buffer, encrypted })
    }

    /// The footer bytes, excluding the trailing metadata length and magic bytes.
    fn metadata(&self) -> Buffer<u8> {
        self.buffer
            .clone()
            .sliced(..self.buffer.len() - FOOTER_SIZE as usize)
    }

    /// Get the footer bytes, or an error if the footer is encrypted.
    fn into_plaintext(self) -> ParquetResult<Buffer<u8>> {
        if self.encrypted {
            return Err(encryption_err!(
                "Parquet file has an encrypted footer but decryption properties were not provided"
            ));
        }
        Ok(self.buffer)
    }
}

/// Check the magic bytes at the end of a Parquet file, and return whether the footer is encrypted.
fn is_encrypted_footer(file_magic: &[u8]) -> ParquetResult<bool> {
    if file_magic == PARQUET_MAGIC {
        Ok(false)
    } else if file_magic == ENCRYPTED_PARQUET_MAGIC {
        Ok(true)
    } else {
        Err(ParquetError::oos("The file must end with PAR1 or PARE"))
    }
}

/// Fetch the trailing footer bytes from a [`Read`] + [`Seek`]. Returns a
/// [`Buffer<u8>`] (not a `Vec<u8>`) because [`FileMetadata`] holds the buffer
/// for the lifetime of the metadata; column-chunk statistics store
/// `ByteRange`s into it instead of allocating per-stat byte vecs.
fn fetch_footer_buf<R: Read + Seek>(reader: &mut R, file_size: u64) -> ParquetResult<FooterBuffer> {
    if file_size < HEADER_SIZE + FOOTER_SIZE {
        return Err(ParquetError::oos(
            "A Parquet file must contain a header and footer with at least 12 bytes",
        ));
    }

    // Read and cache up to DEFAULT_FOOTER_READ_SIZE bytes from the end.
    let default_end_len = min(DEFAULT_FOOTER_READ_SIZE, file_size) as usize;
    reader.seek(SeekFrom::End(-(default_end_len as i64)))?;

    let mut buffer = vec![];
    buffer.try_reserve(default_end_len)?;
    reader
        .take(default_end_len as u64)
        .read_to_end(&mut buffer)?;

    // Check this is indeed a parquet file.
    let encrypted = is_encrypted_footer(&buffer[default_end_len - 4..])?;

    let metadata_len = metadata_len(&buffer) as u64;
    let footer_len = FOOTER_SIZE + metadata_len;
    if footer_len > file_size {
        return Err(ParquetError::oos(
            "The footer size must be smaller or equal to the file's size",
        ));
    }

    // Both branches end with a zero-copy move from `Vec<u8>` into `Buffer`.
    let footer_buf: Buffer<u8> = if (footer_len as usize) <= buffer.len() {
        // Full footer already in the prefetched bytes; slice the tail.
        let remaining = buffer.len() - footer_len as usize;
        Buffer::from_vec(buffer).sliced(remaining..)
    } else {
        // Prefetch wasn't long enough; re-read the whole footer.
        reader.seek(SeekFrom::End(-(footer_len as i64)))?;
        buffer.clear();
        buffer.try_reserve(footer_len as usize)?;
        reader.take(footer_len).read_to_end(&mut buffer)?;
        Buffer::from_vec(buffer)
    };

    Ok(FooterBuffer {
        buffer: footer_buf,
        encrypted,
    })
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;
    use crate::parquet::encryption::test_file_path;

    const FOOTER_KEY: &[u8] = b"0123456789012345";
    const COLUMN_KEY_1: &[u8] = b"1234567890123450";
    const COLUMN_KEY_2: &[u8] = b"1234567890123451";

    fn open_test_file(name: &str) -> File {
        File::open(test_file_path(name)).unwrap()
    }

    fn footer_key_properties(footer_key: &[u8]) -> Arc<FileDecryptionProperties> {
        FileDecryptionProperties::builder(footer_key.to_vec())
            .build()
            .unwrap()
    }

    fn column_key_properties(footer_key: &[u8]) -> Arc<FileDecryptionProperties> {
        FileDecryptionProperties::builder(footer_key.to_vec())
            .with_column_key("double_field", COLUMN_KEY_1.to_vec())
            .with_column_key("float_field", COLUMN_KEY_2.to_vec())
            .build()
            .unwrap()
    }

    /// Check the metadata matches the files written by the Arrow C++ encryption tests.
    fn check_metadata(metadata: &FileMetadata) {
        assert_eq!(metadata.num_rows, 50);
        let row_group_rows: usize = metadata.row_groups.iter().map(|rg| rg.num_rows()).sum();
        assert_eq!(row_group_rows, 50);
        let column_names: Vec<&str> = metadata
            .schema_descr
            .columns()
            .iter()
            .map(|c| c.path_in_schema.last().unwrap().as_str())
            .collect();
        assert_eq!(
            column_names,
            [
                "boolean_field",
                "int32_field",
                "int64_field",
                "int96_field",
                "float_field",
                "double_field",
                "ba_field",
                "flba_field",
            ]
        );
    }

    #[test]
    fn read_encrypted_footer() {
        let mut file = open_test_file("uniform_encryption.parquet.encrypted");
        let metadata = read_metadata_with_decryption(
            &mut file,
            Some(&footer_key_properties(FOOTER_KEY)),
            None,
        )
        .unwrap();
        check_metadata(&metadata);
        assert!(metadata.decryptor.is_some());
    }

    #[test]
    fn read_encrypted_footer_with_file_size() {
        let mut file = open_test_file("uniform_encryption.parquet.encrypted");
        let file_size = file.metadata().unwrap().len();
        let metadata = read_metadata_with_decryption(
            &mut file,
            Some(&footer_key_properties(FOOTER_KEY)),
            Some(file_size),
        )
        .unwrap();
        check_metadata(&metadata);
    }

    #[test]
    fn read_encrypted_footer_with_wrong_key() {
        let mut file = open_test_file("uniform_encryption.parquet.encrypted");
        let result = read_metadata_with_decryption(
            &mut file,
            Some(&footer_key_properties(COLUMN_KEY_1)),
            None,
        );
        assert!(matches!(result, Err(ParquetError::Encryption(_))));
    }

    #[test]
    fn read_encrypted_footer_without_decryption_properties() {
        let mut file = open_test_file("uniform_encryption.parquet.encrypted");
        let result = read_metadata(&mut file);
        assert!(matches!(result, Err(ParquetError::Encryption(_))));
    }

    #[test]
    fn read_plaintext_footer() {
        let mut file = open_test_file("encrypt_columns_plaintext_footer.parquet.encrypted");
        let metadata = read_metadata_with_decryption(
            &mut file,
            Some(&column_key_properties(FOOTER_KEY)),
            None,
        )
        .unwrap();
        check_metadata(&metadata);
        assert!(metadata.decryptor.is_some());
    }

    #[test]
    fn read_plaintext_footer_with_wrong_key() {
        // The footer key is used to verify the footer signature.
        let mut file = open_test_file("encrypt_columns_plaintext_footer.parquet.encrypted");
        let result = read_metadata_with_decryption(
            &mut file,
            Some(&column_key_properties(COLUMN_KEY_1)),
            None,
        );
        assert!(matches!(result, Err(ParquetError::Encryption(_))));
    }

    #[test]
    fn read_plaintext_footer_with_wrong_key_without_verification() {
        let mut file = open_test_file("encrypt_columns_plaintext_footer.parquet.encrypted");
        let decryption_properties = FileDecryptionProperties::builder(COLUMN_KEY_1.to_vec())
            .disable_footer_signature_verification()
            .build()
            .unwrap();
        let metadata =
            read_metadata_with_decryption(&mut file, Some(&decryption_properties), None).unwrap();
        check_metadata(&metadata);
    }

    #[test]
    fn read_plaintext_footer_without_decryption_properties() {
        // The footer can be read without keys, but there's no decryptor for encrypted columns.
        let mut file = open_test_file("encrypt_columns_plaintext_footer.parquet.encrypted");
        let metadata = read_metadata(&mut file).unwrap();
        check_metadata(&metadata);
        assert!(metadata.decryptor.is_none());
    }

    /// The footer bytes of a test file, including the trailing metadata length and magic.
    fn footer_bytes(name: &str) -> Buffer<u8> {
        let file = std::fs::read(test_file_path(name)).unwrap();
        let footer_len = metadata_len(&file) as usize + FOOTER_SIZE as usize;
        Buffer::from_vec(file[file.len() - footer_len..].to_vec())
    }

    #[test]
    fn deserialize_encrypted_footer() {
        let footer = footer_bytes("uniform_encryption.parquet.encrypted");
        let metadata =
            deserialize_metadata_with_decryption(footer, Some(&footer_key_properties(FOOTER_KEY)))
                .unwrap();
        check_metadata(&metadata);
        assert!(metadata.decryptor.is_some());
    }

    #[test]
    fn deserialize_encrypted_footer_without_decryption_properties() {
        let footer = footer_bytes("uniform_encryption.parquet.encrypted");
        let result = deserialize_metadata_with_decryption(footer, None);
        assert!(matches!(result, Err(ParquetError::Encryption(_))));
    }

    #[test]
    fn deserialize_plaintext_footer() {
        let footer = footer_bytes("encrypt_columns_plaintext_footer.parquet.encrypted");
        let metadata =
            deserialize_metadata_with_decryption(footer, Some(&column_key_properties(FOOTER_KEY)))
                .unwrap();
        check_metadata(&metadata);
        assert!(metadata.decryptor.is_some());
    }

    #[test]
    fn deserialize_footer_with_invalid_magic() {
        let footer = Buffer::from_vec(vec![0, 0, 0, 0, b'P', b'A', b'R', b'X']);
        let result = deserialize_metadata_with_decryption(footer, None);
        assert!(matches!(result, Err(ParquetError::OutOfSpec(_))));
    }

    /// Read the metadata of a test file with the given decryption properties.
    fn read_test_file_metadata(
        name: &str,
        decryption_properties: Option<&Arc<FileDecryptionProperties>>,
    ) -> ParquetResult<FileMetadata> {
        read_metadata_with_decryption(&mut open_test_file(name), decryption_properties, None)
    }

    /// The (min, max) statistics of a DOUBLE column in the first row group, if present.
    fn double_column_bounds(metadata: &FileMetadata, name: &str) -> Option<(f64, f64)> {
        let column = metadata.row_groups[0]
            .parquet_columns()
            .iter()
            .find(|c| c.descriptor().path_in_schema[0] == name)
            .unwrap();
        assert_eq!(column.num_values().unwrap(), metadata.num_rows as i64);
        let bounds = column.raw_bounds(&metadata.footer_buf)?;
        let decode = |v: &[u8]| f64::from_le_bytes(v.try_into().unwrap());
        Some((decode(bounds.min?), decode(bounds.max?)))
    }

    #[test]
    fn read_encrypted_column_metadata() {
        for name in [
            "encrypt_columns_and_footer.parquet.encrypted",
            "encrypt_columns_and_footer_aad.parquet.encrypted",
        ] {
            let metadata =
                read_test_file_metadata(name, Some(&column_key_properties(FOOTER_KEY))).unwrap();
            check_metadata(&metadata);
            // Statistics are read from the decrypted column metadata.
            assert_eq!(
                double_column_bounds(&metadata, "double_field"),
                Some((0.0, 49.0 * 1.1111111))
            );
        }
    }

    #[test]
    fn read_encrypted_column_metadata_with_bloom_filters() {
        // This file has a different schema and data to the other test files.
        let metadata = read_test_file_metadata(
            "encrypt_columns_and_footer_bloom_filter.parquet.encrypted",
            Some(&column_key_properties(FOOTER_KEY)),
        )
        .unwrap();
        assert_eq!(metadata.num_rows, 2000);
        assert_eq!(
            double_column_bounds(&metadata, "double_field"),
            Some((0.5, 1999.5))
        );
    }

    #[test]
    fn read_encrypted_column_metadata_with_aad_prefix() {
        let name = "encrypt_columns_and_footer_disable_aad_storage.parquet.encrypted";
        let without_prefix =
            read_test_file_metadata(name, Some(&column_key_properties(FOOTER_KEY)));
        assert!(matches!(without_prefix, Err(ParquetError::Encryption(_))));

        let decryption_properties = FileDecryptionProperties::builder(FOOTER_KEY.to_vec())
            .with_column_key("double_field", COLUMN_KEY_1.to_vec())
            .with_column_key("float_field", COLUMN_KEY_2.to_vec())
            .with_aad_prefix(b"tester".to_vec())
            .build()
            .unwrap();
        let metadata = read_test_file_metadata(name, Some(&decryption_properties)).unwrap();
        check_metadata(&metadata);
    }

    #[test]
    fn read_encrypted_column_metadata_without_column_keys() {
        // Metadata can be read without column keys, but columns encrypted with
        // an unavailable column key have no column metadata.
        let metadata = read_test_file_metadata(
            "encrypt_columns_and_footer.parquet.encrypted",
            Some(&footer_key_properties(FOOTER_KEY)),
        )
        .unwrap();
        check_metadata(&metadata);

        for column in metadata.row_groups[0].parquet_columns() {
            let name = column.descriptor().path_in_schema[0].as_str();
            if name == "double_field" || name == "float_field" {
                // Accessing the column metadata errors, but statistics are just missing.
                assert!(column.raw_bounds(&metadata.footer_buf).is_none());
                assert!(column.null_count().is_none());
                let Err(ParquetError::Encryption(message)) = column.byte_range() else {
                    panic!("expected an encryption error");
                };
                assert!(
                    message.contains(&format!("Metadata for column '{name}' is encrypted")),
                    "{message}"
                );
                assert!(column.num_values().is_err());
            } else {
                column.byte_range().unwrap();
                column.num_values().unwrap();
            }
        }
    }

    #[test]
    fn read_encrypted_column_metadata_with_some_column_keys() {
        let decryption_properties = FileDecryptionProperties::builder(FOOTER_KEY.to_vec())
            .with_column_key("double_field", COLUMN_KEY_1.to_vec())
            .build()
            .unwrap();
        let metadata = read_test_file_metadata(
            "encrypt_columns_and_footer.parquet.encrypted",
            Some(&decryption_properties),
        )
        .unwrap();

        assert_eq!(
            double_column_bounds(&metadata, "double_field"),
            Some((0.0, 49.0 * 1.1111111))
        );
        let float_column = metadata.row_groups[0]
            .parquet_columns()
            .iter()
            .find(|c| c.descriptor().path_in_schema[0] == "float_field")
            .unwrap();
        assert!(float_column.byte_range().is_err());
    }

    #[test]
    fn read_encrypted_column_metadata_with_wrong_column_key() {
        let decryption_properties = FileDecryptionProperties::builder(FOOTER_KEY.to_vec())
            .with_column_key("double_field", COLUMN_KEY_2.to_vec())
            .with_column_key("float_field", COLUMN_KEY_1.to_vec())
            .build()
            .unwrap();
        let result = read_test_file_metadata(
            "encrypt_columns_and_footer.parquet.encrypted",
            Some(&decryption_properties),
        );
        let Err(ParquetError::Encryption(message)) = result else {
            panic!("expected an encryption error, got {result:?}");
        };
        assert!(message.contains("Unable to decrypt metadata"), "{message}");
    }

    #[test]
    fn read_plaintext_footer_column_metadata() {
        let name = "encrypt_columns_plaintext_footer.parquet.encrypted";

        // Without keys, encrypted columns only have stripped metadata without statistics.
        let metadata = read_test_file_metadata(name, None).unwrap();
        assert_eq!(double_column_bounds(&metadata, "double_field"), None);

        // With keys, the full column metadata is decrypted.
        let metadata =
            read_test_file_metadata(name, Some(&column_key_properties(FOOTER_KEY))).unwrap();
        assert_eq!(
            double_column_bounds(&metadata, "double_field"),
            Some((0.0, 49.0 * 1.1111111))
        );
    }
}
