use std::cmp::min;
use std::io::{Read, Seek, SeekFrom};

use parquet_format_safe::thrift::protocol::TCompactInputProtocol;
use parquet_format_safe::FileMetaData as TFileMetaData;

use super::super::metadata::FileMetaData;
use super::super::{DEFAULT_FOOTER_READ_SIZE, FOOTER_SIZE, HEADER_SIZE, PARQUET_MAGIC};
use crate::parquet::error::{ParquetError, ParquetResult};

pub(super) fn metadata_len(buffer: &[u8], len: usize) -> i32 {
    i32::from_le_bytes(buffer[len - 8..len - 4].try_into().unwrap())
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

/// Reads a [`FileMetaData`] from the reader, located at the end of the file.
pub fn read_metadata<R: Read + Seek>(reader: &mut R) -> ParquetResult<FileMetaData> {
    // check file is large enough to hold footer
    let file_size = stream_len(reader)?;
    read_metadata_with_size(reader, file_size)
}

/// Reads a [`FileMetaData`] from the reader, located at the end of the file, with known file size.
pub fn read_metadata_with_size<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
) -> ParquetResult<FileMetaData> {
    if file_size < HEADER_SIZE + FOOTER_SIZE {
        return Err(ParquetError::oos(
            "A parquet file must contain a header and footer with at least 12 bytes",
        ));
    }

    // read and cache up to DEFAULT_FOOTER_READ_SIZE bytes from the end and process the footer
    let default_end_len = min(DEFAULT_FOOTER_READ_SIZE, file_size) as usize;
    reader.seek(SeekFrom::End(-(default_end_len as i64)))?;

    let mut buffer = Vec::with_capacity(default_end_len);
    reader
        .by_ref()
        .take(default_end_len as u64)
        .read_to_end(&mut buffer)?;

    // check this is indeed a parquet file
    if buffer[default_end_len - 4..] != PARQUET_MAGIC {
        return Err(ParquetError::oos("The file must end with PAR1"));
    }

    let metadata_len = metadata_len(&buffer, default_end_len);

    let metadata_len: u64 = metadata_len.try_into()?;

    let footer_len = FOOTER_SIZE + metadata_len;
    if footer_len > file_size {
        return Err(ParquetError::oos(
            "The footer size must be smaller or equal to the file's size",
        ));
    }

    let reader: &[u8] = if (footer_len as usize) < buffer.len() {
        // the whole metadata is in the bytes we already read
        let remaining = buffer.len() - footer_len as usize;
        &buffer[remaining..]
    } else {
        // the end of file read by default is not long enough, read again including the metadata.
        reader.seek(SeekFrom::End(-(footer_len as i64)))?;

        buffer.clear();
        buffer.try_reserve(footer_len as usize)?;
        reader.take(footer_len).read_to_end(&mut buffer)?;

        &buffer
    };

    // a highly nested but sparse struct could result in many allocations
    let max_size = reader.len() * 2 + 1024;

    deserialize_metadata(reader, max_size)
}

/// Parse loaded metadata bytes
pub fn deserialize_metadata<R: Read>(reader: R, max_size: usize) -> ParquetResult<FileMetaData> {
    let mut prot = TCompactInputProtocol::new(reader, max_size);
    let metadata = TFileMetaData::read_from_in_protocol(&mut prot)?;

    FileMetaData::try_from_thrift(metadata)
}
