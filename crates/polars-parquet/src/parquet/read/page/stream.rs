use std::io::{Cursor, SeekFrom};

use async_stream::try_stream;
use futures::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, Stream};
use polars_buffer::Buffer;
use polars_parquet_format::thrift::protocol::TCompactInputStreamProtocol;

use super::reader::{PageMetaData, decrypt_page_data, finish_page};
use crate::parquet::compression::Compression;
use crate::parquet::encryption::ciphers::{NONCE_LEN, SIZE_LEN, TAG_LEN};
use crate::parquet::encryption::decrypt::CryptoContext;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnChunkMetadata, Descriptor};
use crate::parquet::page::{CompressedPage, DataPageHeader, ParquetPageHeader};
use crate::parquet::parquet_bridge::{Encoding, PageType};

struct StreamPageMetaData {
    num_values: i64,
    compression: Compression,
    descriptor: Descriptor,
    has_dictionary_page: bool,
    crypto_context: Option<CryptoContext>,
}

/// Returns a stream of compressed data pages
pub async fn get_page_stream<'a, RR: AsyncRead + Unpin + Send + AsyncSeek>(
    column_metadata: &'a ColumnChunkMetadata,
    reader: &'a mut RR,
    scratch: Vec<u8>,
    max_page_size: usize,
) -> ParquetResult<impl Stream<Item = ParquetResult<CompressedPage>> + 'a> {
    let page_metadata: PageMetaData = column_metadata.into();
    let column_start = page_metadata.column_start;
    reader.seek(SeekFrom::Start(column_start)).await?;
    Ok(_get_page_stream(
        reader,
        scratch,
        max_page_size,
        StreamPageMetaData {
            num_values: page_metadata.num_values,
            compression: page_metadata.compression,
            descriptor: page_metadata.descriptor,
            has_dictionary_page: page_metadata.dictionary_page_offset == Some(column_start),
            crypto_context: column_metadata.crypto_context().cloned(),
        },
    ))
}

/// Returns a stream of compressed data pages from a reader that begins at the start of the column
pub async fn get_page_stream_from_column_start<'a, R: AsyncRead + Unpin + Send>(
    column_metadata: &'a ColumnChunkMetadata,
    reader: &'a mut R,
    scratch: Vec<u8>,
    max_header_size: usize,
) -> ParquetResult<impl Stream<Item = ParquetResult<CompressedPage>> + 'a> {
    let page_metadata: PageMetaData = column_metadata.into();
    Ok(_get_page_stream(
        reader,
        scratch,
        max_header_size,
        StreamPageMetaData {
            num_values: page_metadata.num_values,
            compression: page_metadata.compression,
            descriptor: page_metadata.descriptor,
            has_dictionary_page: page_metadata.dictionary_page_offset
                == Some(page_metadata.column_start),
            crypto_context: column_metadata.crypto_context().cloned(),
        },
    ))
}

/// Returns a stream of compressed data pages with [`PageMetaData`]
fn _get_page_stream<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
    mut scratch: Vec<u8>,
    max_page_size: usize,
    page_metadata: StreamPageMetaData,
) -> impl Stream<Item = ParquetResult<CompressedPage>> + '_ {
    let mut seen_values = 0i64;
    let mut page_ordinal = 0usize;
    let mut dictionary_page_read = false;
    try_stream! {
        while seen_values < page_metadata.num_values {
            let dictionary_page = page_metadata.has_dictionary_page && !dictionary_page_read;
            // the header
            let page_header = read_page_header(
                reader,
                max_page_size,
                page_metadata.crypto_context.as_ref(),
                page_ordinal,
                dictionary_page,
            ).await?;

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

            let data = decrypt_page_data(
                Buffer::from_vec(std::mem::take(&mut scratch)),
                page_metadata.crypto_context.as_ref(),
                page_ordinal,
                dictionary_page,
            )?;

            if !dictionary_page {
                page_ordinal += 1;
            } else {
                dictionary_page_read = true;
            }

            yield finish_page(
                page_header,
                data,
                page_metadata.compression,
                &page_metadata.descriptor,
            )?;
        }
    }
}

/// Reads Page header from Thrift.
async fn read_page_header<R: AsyncRead + Unpin + Send>(
    reader: &mut R,
    max_page_size: usize,
    crypto_context: Option<&CryptoContext>,
    page_ordinal: usize,
    dictionary_page: bool,
) -> ParquetResult<ParquetPageHeader> {
    let Some(crypto_context) = crypto_context else {
        let mut prot = TCompactInputStreamProtocol::new(reader, max_page_size);
        return Ok(ParquetPageHeader::stream_from_in_protocol(&mut prot).await?);
    };

    let page_crypto_context = if dictionary_page {
        crypto_context.for_dictionary_page()
    } else {
        crypto_context.with_page_ordinal(page_ordinal)
    };
    let aad = page_crypto_context.create_page_header_aad()?;
    let mut len_bytes = [0; SIZE_LEN];
    reader.read_exact(&mut len_bytes).await?;
    let ciphertext_len = u32::from_le_bytes(len_bytes) as usize;
    let max_ciphertext_len = max_page_size.saturating_add(NONCE_LEN + TAG_LEN);
    if ciphertext_len > max_ciphertext_len {
        return Err(ParquetError::WouldOverAllocate);
    }
    let mut ciphertext = vec![0; SIZE_LEN + ciphertext_len];
    ciphertext[..SIZE_LEN].copy_from_slice(&len_bytes);
    reader.read_exact(&mut ciphertext[SIZE_LEN..]).await?;
    let decrypted = page_crypto_context
        .metadata_decryptor()
        .decrypt(&ciphertext, &aad)
        .map_err(|_| ParquetError::oos("failed to decrypt parquet page header"))?;

    let mut cursor = Cursor::new(decrypted);
    let mut prot = polars_parquet_format::thrift::protocol::TCompactInputProtocol::new(
        &mut cursor,
        max_page_size,
    );
    Ok(ParquetPageHeader::read_from_in_protocol(&mut prot)?)
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
