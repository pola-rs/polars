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
    read_metadata_with_decryption(reader, None, None)
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

/// Parse loaded metadata bytes via the hand-written Thrift compact decoder,
/// using the provided decryption properties if the file is encrypted.
///
/// `footer` must be a [`Buffer<u8>`] because [`FileMetadata`] holds the buffer
/// for the lifetime of the metadata; column-chunk statistics store
/// `ByteRange`s into it instead of allocating per-stat byte vecs.
/// `footer` must include the trailing metadata length and magic bytes, which are used
/// to determine whether the footer is encrypted.
///
/// Files with an encrypted footer require the decryption properties to read the metadata.
/// Files with a plaintext footer may still have encrypted columns. If decryption properties
/// are provided, the footer signature is verified unless disabled in the decryption properties.
pub fn deserialize_metadata(
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
        let compact = decode_file_metadata(footer.into_plaintext()?)?;
        return FileMetadata::from_compact(compact, None);
    };

    if footer.encrypted {
        let (footer, decryptor) = decrypt_footer(&footer, decryption_properties)?;
        let compact = decode_file_metadata(footer)?;
        FileMetadata::from_compact(compact, Some(Arc::new(decryptor)))
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
        FileMetadata::from_compact(compact, decryptor)
    }
}

/// Decrypt an encrypted footer, returning the plaintext footer and the file decryptor.
fn decrypt_footer(
    footer: &FooterBuffer,
    decryption_properties: &Arc<FileDecryptionProperties>,
) -> ParquetResult<(Buffer<u8>, FileDecryptor)> {
    debug_assert!(footer.encrypted);
    // First read the FileCryptoMetaData, which comes before the encrypted footer and is
    // needed to decrypt the footer.
    let (crypto_metadata, encrypted_footer) = deserialize_file_crypto_metadata(footer.metadata())?;
    let decryptor = FileDecryptor::from_encryption_algorithm(
        decryption_properties,
        crypto_metadata.encryption_algorithm,
        crypto_metadata.key_metadata.as_deref(),
    )?;
    let footer = Buffer::from_vec(decryptor.decrypt_footer(&encrypted_footer)?);
    Ok((footer, decryptor))
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
///
/// `footer` must include the trailing metadata length and magic bytes.
/// An encrypted footer is decrypted first, which requires the decryption properties.
/// The footer signature of a plaintext footer isn't verified.
pub fn deserialize_num_rows(
    footer: Buffer<u8>,
    decryption_properties: Option<&Arc<FileDecryptionProperties>>,
) -> ParquetResult<i64> {
    decode_footer_num_rows(FooterBuffer::try_new(footer)?, decryption_properties)
}

/// Sync variant of [`deserialize_num_rows`] that owns the reader.
pub fn read_num_rows<R: Read + Seek>(
    reader: &mut R,
    decryption_properties: Option<&Arc<FileDecryptionProperties>>,
) -> ParquetResult<i64> {
    let file_size = stream_len(reader)?;
    let footer = fetch_footer_buf(reader, file_size)?;
    decode_footer_num_rows(footer, decryption_properties)
}

fn decode_footer_num_rows(
    footer: FooterBuffer,
    decryption_properties: Option<&Arc<FileDecryptionProperties>>,
) -> ParquetResult<i64> {
    match decryption_properties {
        Some(decryption_properties) if footer.encrypted => {
            let (footer, _) = decrypt_footer(&footer, decryption_properties)?;
            decode_num_rows(footer)
        },
        _ => decode_num_rows(footer.into_plaintext()?),
    }
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
    use super::*;

    #[test]
    fn deserialize_footer_with_invalid_magic() {
        let footer = Buffer::from_vec(vec![0, 0, 0, 0, b'P', b'A', b'R', b'X']);
        let result = deserialize_metadata(footer, None);
        assert!(matches!(result, Err(ParquetError::OutOfSpec(_))));
    }
}
