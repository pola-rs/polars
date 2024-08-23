use std::io::SeekFrom;

use async_stream::try_stream;
use futures::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, Stream};
use parquet_format_safe::thrift::protocol::TCompactInputStreamProtocol;
use polars_utils::mmap::MemSlice;

use super::reader::{finish_page, PageMetaData};
use crate::parquet::compression::Compression;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnChunkMetaData, Descriptor};
use crate::parquet::page::{CompressedPage, DataPageHeader, ParquetPageHeader};
use crate::parquet::parquet_bridge::{Encoding, PageType};

/// Returns a stream of compressed data pages
pub async fn get_page_stream<'a, RR: AsyncRead + Unpin + Send + AsyncSeek>(
    column_metadata: &'a ColumnChunkMetaData,
    reader: &'a mut RR,
    scratch: Vec<u8>,
    max_page_size: usize,
) -> ParquetResult<impl Stream<Item = ParquetResult<CompressedPage>> + 'a> {
    get_page_stream_with_page_meta(column_metadata.into(), reader, scratch, max_page_size).await
}

/// Returns a stream of compressed data pages from a reader that begins at the start of the column
pub async fn get_page_stream_from_column_start<'a, R: AsyncRead + Unpin + Send>(
    column_metadata: &'a ColumnChunkMetaData,
    reader: &'a mut R,
    scratch: Vec<u8>,
    max_header_size: usize,
) -> ParquetResult<impl Stream<Item = ParquetResult<CompressedPage>> + 'a> {
    let page_metadata: PageMetaData = column_metadata.into();
    Ok(_get_page_stream(
        reader,
        page_metadata.num_values,
        page_metadata.compression,
        page_metadata.descriptor,
        scratch,
        max_header_size,
    ))
}

/// Returns a stream of compressed data pages with [`PageMetaData`]
pub async fn get_page_stream_with_page_meta<RR: AsyncRead + Unpin + Send + AsyncSeek>(
    page_metadata: PageMetaData,
    reader: &mut RR,
    scratch: Vec<u8>,
    max_page_size: usize,
) -> ParquetResult<impl Stream<Item = ParquetResult<CompressedPage>> + '_> {
    let column_start = page_metadata.column_start;
    reader.seek(SeekFrom::Start(column_start)).await?;
    Ok(_get_page_stream(
        reader,
        page_metadata.num_values,
        page_metadata.compression,
        page_metadata.descriptor,
        scratch,
        max_page_size,
    ))
}

fn _get_page_stream<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
    total_num_values: i64,
    compression: Compression,
    descriptor: Descriptor,
    mut scratch: Vec<u8>,
    max_page_size: usize,
) -> impl Stream<Item = ParquetResult<CompressedPage>> + '_ {
    let mut seen_values = 0i64;
    try_stream! {
        while seen_values < total_num_values {
            // the header
            let page_header = read_page_header(reader, max_page_size).await?;

            let data_header = get_page_header(&page_header)?;
            seen_values += data_header.as_ref().map(|x| x.num_values() as i64).unwrap_or_default();

            let read_size: usize = page_header.compressed_page_size.try_into()?;

            if read_size > max_page_size {
                Err(ParquetError::WouldOverAllocate)?
            }

            // followed by the buffer
            scratch.clear();
            scratch.try_reserve(read_size)?;
            let bytes_read = reader
                .take(read_size as u64)
                .read_to_end(&mut scratch).await?;

            if bytes_read != read_size {
                Err(ParquetError::oos(
                    "The page header reported the wrong page size",
                ))?
            }

            yield finish_page(
                page_header,
                MemSlice::from_vec(std::mem::take(&mut scratch)),
                compression,
                &descriptor,
            )?;
        }
    }
}

/// Reads Page header from Thrift.
async fn read_page_header<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
    max_page_size: usize,
) -> ParquetResult<ParquetPageHeader> {
    let mut prot = TCompactInputStreamProtocol::new(reader, max_page_size);
    let page_header = ParquetPageHeader::stream_from_in_protocol(&mut prot).await?;
    Ok(page_header)
}

pub(super) fn get_page_header(header: &ParquetPageHeader) -> ParquetResult<Option<DataPageHeader>> {
    let type_ = header.type_.try_into()?;
    Ok(match type_ {
        PageType::DataPage => {
            let header = header.data_page_header.clone().ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a v1 data page but the v1 header is empty",
                )
            })?;
            let _: Encoding = header.encoding.try_into()?;
            let _: Encoding = header.repetition_level_encoding.try_into()?;
            let _: Encoding = header.definition_level_encoding.try_into()?;

            Some(DataPageHeader::V1(header))
        },
        PageType::DataPageV2 => {
            let header = header.data_page_header_v2.clone().ok_or_else(|| {
                ParquetError::oos(
                    "The page header type is a v1 data page but the v1 header is empty",
                )
            })?;
            let _: Encoding = header.encoding.try_into()?;
            Some(DataPageHeader::V2(header))
        },
        _ => None,
    })
}
