use std::io::SeekFrom;

use futures::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt};

use super::super::metadata::FileMetaData;
use super::super::{DEFAULT_FOOTER_READ_SIZE, FOOTER_SIZE, PARQUET_MAGIC};
use super::metadata::{deserialize_metadata, metadata_len};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::HEADER_SIZE;

async fn stream_len(
    seek: &mut (impl AsyncSeek + std::marker::Unpin),
) -> std::result::Result<u64, std::io::Error> {
    let old_pos = seek.seek(SeekFrom::Current(0)).await?;
    let len = seek.seek(SeekFrom::End(0)).await?;

    // Avoid seeking a third time when we were already at the end of the
    // stream. The branch is usually way cheaper than a seek operation.
    if old_pos != len {
        seek.seek(SeekFrom::Start(old_pos)).await?;
    }

    Ok(len)
}

/// Asynchronously reads the files' metadata
pub async fn read_metadata<R: AsyncRead + AsyncSeek + Send + std::marker::Unpin>(
    reader: &mut R,
) -> ParquetResult<FileMetaData> {
    let file_size = stream_len(reader).await?;

    if file_size < HEADER_SIZE + FOOTER_SIZE {
        return Err(ParquetError::oos(
            "A parquet file must contain a header and footer with at least 12 bytes",
        ));
    }

    // read and cache up to DEFAULT_FOOTER_READ_SIZE bytes from the end and process the footer
    let default_end_len = std::cmp::min(DEFAULT_FOOTER_READ_SIZE, file_size) as usize;
    reader
        .seek(SeekFrom::End(-(default_end_len as i64)))
        .await?;

    let mut buffer = vec![];
    buffer.try_reserve(default_end_len)?;
    reader
        .take(default_end_len as u64)
        .read_to_end(&mut buffer)
        .await?;

    // check this is indeed a parquet file
    if buffer[default_end_len - 4..] != PARQUET_MAGIC {
        return Err(ParquetError::oos("Invalid Parquet file. Corrupt footer"));
    }

    let metadata_len = metadata_len(&buffer, default_end_len);
    let metadata_len: u64 = metadata_len.try_into()?;

    let footer_len = FOOTER_SIZE + metadata_len;
    if footer_len > file_size {
        return Err(ParquetError::oos(
            "The footer size must be smaller or equal to the file's size",
        ));
    }

    let reader = if (footer_len as usize) < buffer.len() {
        // the whole metadata is in the bytes we already read
        let remaining = buffer.len() - footer_len as usize;
        &buffer[remaining..]
    } else {
        // the end of file read by default is not long enough, read again including the metadata.
        reader.seek(SeekFrom::End(-(footer_len as i64))).await?;

        buffer.clear();
        buffer.try_reserve(footer_len as usize)?;
        reader
            .take(footer_len as u64)
            .read_to_end(&mut buffer)
            .await?;

        &buffer
    };

    // a highly nested but sparse struct could result in many allocations
    let max_size = reader.len() * 2 + 1024;

    deserialize_metadata(reader, max_size)
}
