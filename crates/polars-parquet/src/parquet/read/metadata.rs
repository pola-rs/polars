use std::cmp::min;
use std::io::{Read, Seek, SeekFrom};

use polars_buffer::Buffer;

use super::super::metadata::{FileMetadata, SchemaDescriptor};
use super::super::{DEFAULT_FOOTER_READ_SIZE, FOOTER_SIZE, HEADER_SIZE, PARQUET_MAGIC};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::handwritten_thrift::{
    decode_file_metadata, decode_file_metadata_skip_schema, decode_num_rows,
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
    deserialize_metadata(footer)
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

/// Parse `footer` with thrift field 2 (schema) skipped, reusing
/// `schema_descr`. See [`deserialize_metadata`] for the buffer-lifetime
/// contract and [`crate::parquet::handwritten_thrift::decode_file_metadata_skip_schema`]
/// for the decode rationale.
pub fn deserialize_metadata_with_shared_schema(
    footer: Buffer<u8>,
    schema_descr: SchemaDescriptor,
) -> ParquetResult<FileMetadata> {
    let compact = decode_file_metadata_skip_schema(footer)?;
    FileMetadata::from_compact_with_schema_descr(compact, schema_descr)
}

/// Sync variant of [`deserialize_metadata_with_shared_schema`] that owns the
/// reader. Fetches the footer the same way [`read_metadata`] does, then
/// parses with the schema field skipped.
pub fn read_metadata_with_shared_schema<R: Read + Seek>(
    reader: &mut R,
    schema_descr: SchemaDescriptor,
) -> ParquetResult<FileMetadata> {
    let file_size = stream_len(reader)?;
    read_metadata_with_shared_schema_with_size(reader, file_size, schema_descr)
}

/// As [`read_metadata_with_shared_schema`] but with a pre-fetched file size.
pub(crate) fn read_metadata_with_shared_schema_with_size<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    schema_descr: SchemaDescriptor,
) -> ParquetResult<FileMetadata> {
    let footer = fetch_footer_buf(reader, file_size)?;
    deserialize_metadata_with_shared_schema(footer, schema_descr)
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
    if buffer[default_end_len - 4..] != PARQUET_MAGIC {
        return Err(ParquetError::oos("The file must end with PAR1"));
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
