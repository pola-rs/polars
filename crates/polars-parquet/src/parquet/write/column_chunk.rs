use std::io::Write;

#[cfg(feature = "async")]
use futures::AsyncWrite;
use parquet_format_safe::thrift::protocol::TCompactOutputProtocol;
#[cfg(feature = "async")]
use parquet_format_safe::thrift::protocol::TCompactOutputStreamProtocol;
use parquet_format_safe::{ColumnChunk, ColumnMetaData, Type};
use polars_utils::aliases::PlHashSet;

#[cfg(feature = "async")]
use super::page::write_page_async;
use super::page::{write_page, PageWriteSpec};
use super::statistics::reduce;
use super::DynStreamingIterator;
use crate::parquet::compression::Compression;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::ColumnDescriptor;
use crate::parquet::page::{CompressedPage, PageType};
use crate::parquet::FallibleStreamingIterator;

pub fn write_column_chunk<W, E>(
    writer: &mut W,
    mut offset: u64,
    descriptor: &ColumnDescriptor,
    mut compressed_pages: DynStreamingIterator<'_, CompressedPage, E>,
) -> ParquetResult<(ColumnChunk, Vec<PageWriteSpec>, u64)>
where
    W: Write,
    ParquetError: From<E>,
    E: std::error::Error,
{
    // write every page

    let initial = offset;

    let mut specs = vec![];
    while let Some(compressed_page) = compressed_pages.next()? {
        let spec = write_page(writer, offset, compressed_page)?;
        offset += spec.bytes_written;
        specs.push(spec);
    }
    let mut bytes_written = offset - initial;

    let column_chunk = build_column_chunk(&specs, descriptor)?;

    // write metadata
    let mut protocol = TCompactOutputProtocol::new(writer);
    bytes_written += column_chunk
        .meta_data
        .as_ref()
        .unwrap()
        .write_to_out_protocol(&mut protocol)? as u64;

    Ok((column_chunk, specs, bytes_written))
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn write_column_chunk_async<W, E>(
    writer: &mut W,
    mut offset: u64,
    descriptor: &ColumnDescriptor,
    mut compressed_pages: DynStreamingIterator<'_, CompressedPage, E>,
) -> ParquetResult<(ColumnChunk, Vec<PageWriteSpec>, u64)>
where
    W: AsyncWrite + Unpin + Send,
    ParquetError: From<E>,
    E: std::error::Error,
{
    let initial = offset;
    // write every page
    let mut specs = vec![];
    while let Some(compressed_page) = compressed_pages.next()? {
        let spec = write_page_async(writer, offset, compressed_page).await?;
        offset += spec.bytes_written;
        specs.push(spec);
    }
    let mut bytes_written = offset - initial;

    let column_chunk = build_column_chunk(&specs, descriptor)?;

    // write metadata
    let mut protocol = TCompactOutputStreamProtocol::new(writer);
    bytes_written += column_chunk
        .meta_data
        .as_ref()
        .unwrap()
        .write_to_out_stream_protocol(&mut protocol)
        .await? as u64;

    Ok((column_chunk, specs, bytes_written))
}

fn build_column_chunk(
    specs: &[PageWriteSpec],
    descriptor: &ColumnDescriptor,
) -> ParquetResult<ColumnChunk> {
    // compute stats to build header at the end of the chunk

    let compression = specs
        .iter()
        .map(|spec| spec.compression)
        .collect::<PlHashSet<_>>();
    if compression.len() > 1 {
        return Err(crate::parquet::error::ParquetError::oos(
            "All pages within a column chunk must be compressed with the same codec",
        ));
    }
    let compression = compression
        .into_iter()
        .next()
        .unwrap_or(Compression::Uncompressed);

    // SPEC: the total compressed size is the total compressed size of each page + the header size
    let total_compressed_size = specs
        .iter()
        .map(|x| x.header_size as i64 + x.header.compressed_page_size as i64)
        .sum();
    // SPEC: the total compressed size is the total compressed size of each page + the header size
    let total_uncompressed_size = specs
        .iter()
        .map(|x| x.header_size as i64 + x.header.uncompressed_page_size as i64)
        .sum();
    let data_page_offset = specs.first().map(|spec| spec.offset).unwrap_or(0) as i64;
    let num_values = specs
        .iter()
        .map(|spec| {
            let type_ = spec.header.type_.try_into().unwrap();
            match type_ {
                PageType::DataPage => {
                    spec.header.data_page_header.as_ref().unwrap().num_values as i64
                },
                PageType::DataPageV2 => {
                    spec.header.data_page_header_v2.as_ref().unwrap().num_values as i64
                },
                _ => 0, // only data pages contribute
            }
        })
        .sum();
    let mut encodings = specs
        .iter()
        .flat_map(|spec| {
            let type_ = spec.header.type_.try_into().unwrap();
            match type_ {
                PageType::DataPage => vec![
                    spec.header.data_page_header.as_ref().unwrap().encoding,
                    Encoding::Rle.into(),
                ],
                PageType::DataPageV2 => {
                    vec![
                        spec.header.data_page_header_v2.as_ref().unwrap().encoding,
                        Encoding::Rle.into(),
                    ]
                },
                PageType::DictionaryPage => vec![
                    spec.header
                        .dictionary_page_header
                        .as_ref()
                        .unwrap()
                        .encoding,
                ],
            }
        })
        .collect::<PlHashSet<_>>() // unique
        .into_iter() // to vec
        .collect::<Vec<_>>();

    // Sort the encodings to have deterministic metadata
    encodings.sort();

    let statistics = specs.iter().map(|x| &x.statistics).collect::<Vec<_>>();
    let statistics = reduce(&statistics)?;
    let statistics = statistics.map(|x| x.serialize());

    let (type_, _): (Type, Option<i32>) = descriptor.descriptor.primitive_type.physical_type.into();

    let metadata = ColumnMetaData {
        type_,
        encodings,
        path_in_schema: descriptor.path_in_schema.clone(),
        codec: compression.into(),
        num_values,
        total_uncompressed_size,
        total_compressed_size,
        key_value_metadata: None,
        data_page_offset,
        index_page_offset: None,
        dictionary_page_offset: None,
        statistics,
        encoding_stats: None,
        bloom_filter_offset: None,
    };

    Ok(ColumnChunk {
        file_path: None, // same file for now.
        file_offset: data_page_offset + total_compressed_size,
        meta_data: Some(metadata),
        offset_index_offset: None,
        offset_index_length: None,
        column_index_offset: None,
        column_index_length: None,
        crypto_metadata: None,
        encrypted_column_metadata: None,
    })
}
