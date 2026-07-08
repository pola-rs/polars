use std::cmp::min;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_parquet_format::FileCryptoMetaData;
use polars_parquet_format::thrift::protocol::TCompactInputProtocol;

use super::super::metadata::{FileMetadata, decode_thrift_file_metadata};
use super::super::{
    DEFAULT_FOOTER_READ_SIZE, FOOTER_SIZE, HEADER_SIZE, PARQUET_ENCRYPTED_MAGIC, PARQUET_MAGIC,
};
use crate::parquet::encryption::decrypt::FileDecryptionProperties;
use crate::parquet::encryption::modules::create_footer_aad;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::handwritten_thrift::{decode_file_metadata, decode_num_rows};

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

/// Reads a [`FileMetadata`] from the reader with parquet decryption properties.
pub fn read_metadata_with_decryption<R: Read + Seek>(
    reader: &mut R,
    decryption_properties: Arc<FileDecryptionProperties>,
) -> ParquetResult<FileMetadata> {
    let file_size = stream_len(reader)?;
    read_metadata_with_size_and_decryption(reader, file_size, Some(decryption_properties))
}

/// Reads a [`FileMetadata`] from the reader, located at the end of the file, with known file size.
pub fn read_metadata_with_size<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
) -> ParquetResult<FileMetadata> {
    read_metadata_with_size_and_decryption(reader, file_size, None)
}

pub fn read_metadata_with_size_and_decryption<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    decryption_properties: Option<Arc<FileDecryptionProperties>>,
) -> ParquetResult<FileMetadata> {
    let footer = fetch_footer_buf(reader, file_size)?;
    deserialize_metadata_with_decryption(footer, decryption_properties)
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

pub fn deserialize_metadata_with_decryption(
    footer: Buffer<u8>,
    decryption_properties: Option<Arc<FileDecryptionProperties>>,
) -> ParquetResult<FileMetadata> {
    let encrypted_footer = footer[footer.len() - 4..] == PARQUET_ENCRYPTED_MAGIC;
    if !encrypted_footer && decryption_properties.is_none() {
        return deserialize_metadata(footer);
    }

    let footer_without_tail = &footer[..footer.len() - FOOTER_SIZE as usize];
    let compact = if encrypted_footer {
        let decryption_properties = decryption_properties.ok_or_else(|| {
            ParquetError::InvalidParameter(
                "parquet file has an encrypted footer but no decryption properties were provided"
                    .to_string(),
            )
        })?;
        let mut cursor = Cursor::new(footer_without_tail);
        let mut protocol = TCompactInputProtocol::new(&mut cursor, usize::MAX);
        let file_crypto_metadata = FileCryptoMetaData::read_from_in_protocol(&mut protocol)?;
        let crypto_metadata_len = cursor.position() as usize;
        let file_decryptor = crate::parquet::encryption::decrypt::FileDecryptor::from_algorithm(
            file_crypto_metadata.encryption_algorithm,
            file_crypto_metadata.key_metadata.as_deref(),
            &decryption_properties,
        )?;
        let aad = create_footer_aad(file_decryptor.file_aad())?;
        let footer_decryptor = file_decryptor.get_footer_decryptor()?;
        let decrypted_footer = footer_decryptor
            .decrypt(&footer_without_tail[crypto_metadata_len..], &aad)
            .map_err(|_| ParquetError::oos("failed to decrypt parquet footer"))?;

        decode_thrift_file_metadata(&decrypted_footer, Some(file_decryptor))?
    } else {
        let mut cursor = Cursor::new(footer_without_tail);
        let mut protocol = TCompactInputProtocol::new(&mut cursor, usize::MAX);
        let thrift = polars_parquet_format::FileMetaData::read_from_in_protocol(&mut protocol)?;
        let file_decryptor = match thrift.encryption_algorithm.clone() {
            Some(encryption_algorithm) => {
                let decryption_properties = decryption_properties.ok_or_else(|| {
                    ParquetError::InvalidParameter(
                        "parquet file has encrypted modules but no decryption properties were provided"
                            .to_string(),
                    )
                })?;
                let file_decryptor =
                    crate::parquet::encryption::decrypt::FileDecryptor::from_algorithm(
                        encryption_algorithm,
                        thrift.footer_signing_key_metadata.as_deref(),
                        &decryption_properties,
                    )?;
                if decryption_properties.check_plaintext_footer_integrity() {
                    file_decryptor.verify_plaintext_footer_signature(footer_without_tail)?;
                }
                Some(file_decryptor)
            },
            None => None,
        };
        decode_thrift_file_metadata(footer_without_tail, file_decryptor)?
    };

    FileMetadata::from_compact(compact)
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
    decode_num_rows(footer)
}

/// Fetch the trailing footer bytes from a [`Read`] + [`Seek`]. Returns a
/// [`Buffer<u8>`] (not a `Vec<u8>`) because [`FileMetadata`] holds the buffer
/// for the lifetime of the metadata; column-chunk statistics store
/// `ByteRange`s into it instead of allocating per-stat byte vecs.
fn fetch_footer_buf<R: Read + Seek>(reader: &mut R, file_size: u64) -> ParquetResult<Buffer<u8>> {
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
    let magic = &buffer[default_end_len - 4..];
    if magic != PARQUET_MAGIC && magic != PARQUET_ENCRYPTED_MAGIC {
        return Err(ParquetError::oos("The file must end with PAR1 or PARE"));
    }

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

    Ok(footer_buf)
}
