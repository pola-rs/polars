use std::io::Write;

#[cfg(feature = "async")]
use futures::AsyncWrite;
use parquet_format_safe::{ColumnChunk, RowGroup};

use super::column_chunk::write_column_chunk;
#[cfg(feature = "async")]
use super::column_chunk::write_column_chunk_async;
use super::page::{is_data_page, PageWriteSpec};
use super::{DynIter, DynStreamingIterator};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{ColumnChunkMetaData, ColumnDescriptor};
use crate::parquet::page::CompressedPage;

pub struct ColumnOffsetsMetadata {
    pub dictionary_page_offset: Option<i64>,
    pub data_page_offset: Option<i64>,
}

impl ColumnOffsetsMetadata {
    pub fn from_column_chunk(column_chunk: &ColumnChunk) -> ColumnOffsetsMetadata {
        ColumnOffsetsMetadata {
            dictionary_page_offset: column_chunk
                .meta_data
                .as_ref()
                .map(|meta| meta.dictionary_page_offset)
                .unwrap_or(None),
            data_page_offset: column_chunk
                .meta_data
                .as_ref()
                .map(|meta| meta.data_page_offset),
        }
    }

    pub fn from_column_chunk_metadata(
        column_chunk_metadata: &ColumnChunkMetaData,
    ) -> ColumnOffsetsMetadata {
        ColumnOffsetsMetadata {
            dictionary_page_offset: column_chunk_metadata.dictionary_page_offset(),
            data_page_offset: Some(column_chunk_metadata.data_page_offset()),
        }
    }

    pub fn calc_row_group_file_offset(&self) -> Option<i64> {
        self.dictionary_page_offset
            .filter(|x| *x > 0_i64)
            .or(self.data_page_offset)
    }
}

fn compute_num_rows(columns: &[(ColumnChunk, Vec<PageWriteSpec>)]) -> ParquetResult<i64> {
    columns
        .first()
        .map(|(_, specs)| {
            let mut num_rows = 0;
            specs
                .iter()
                .filter(|x| is_data_page(x))
                .try_for_each(|spec| {
                    num_rows += spec.num_rows as i64;
                    ParquetResult::Ok(())
                })?;
            ParquetResult::Ok(num_rows)
        })
        .unwrap_or(Ok(0))
}

pub fn write_row_group<
    'a,
    W,
    E, // external error any of the iterators may emit
>(
    writer: &mut W,
    mut offset: u64,
    descriptors: &[ColumnDescriptor],
    columns: DynIter<'a, std::result::Result<DynStreamingIterator<'a, CompressedPage, E>, E>>,
    ordinal: usize,
) -> ParquetResult<(RowGroup, Vec<Vec<PageWriteSpec>>, u64)>
where
    W: Write,
    ParquetError: From<E>,
    E: std::error::Error,
{
    let column_iter = descriptors.iter().zip(columns);

    let initial = offset;
    let columns = column_iter
        .map(|(descriptor, page_iter)| {
            let (column, page_specs, size) =
                write_column_chunk(writer, offset, descriptor, page_iter?)?;
            offset += size;
            Ok((column, page_specs))
        })
        .collect::<ParquetResult<Vec<_>>>()?;
    let bytes_written = offset - initial;

    let num_rows = compute_num_rows(&columns)?;

    // compute row group stats
    let file_offset = columns
        .first()
        .map(|(column_chunk, _)| {
            ColumnOffsetsMetadata::from_column_chunk(column_chunk).calc_row_group_file_offset()
        })
        .unwrap_or(None);

    let total_byte_size = columns
        .iter()
        .map(|(c, _)| c.meta_data.as_ref().unwrap().total_uncompressed_size)
        .sum();
    let total_compressed_size = columns
        .iter()
        .map(|(c, _)| c.meta_data.as_ref().unwrap().total_compressed_size)
        .sum();

    let (columns, specs) = columns.into_iter().unzip();

    Ok((
        RowGroup {
            columns,
            total_byte_size,
            num_rows,
            sorting_columns: None,
            file_offset,
            total_compressed_size: Some(total_compressed_size),
            ordinal: ordinal.try_into().ok(),
        },
        specs,
        bytes_written,
    ))
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn write_row_group_async<
    'a,
    W,
    E, // external error any of the iterators may emit
>(
    writer: &mut W,
    mut offset: u64,
    descriptors: &[ColumnDescriptor],
    columns: DynIter<'a, std::result::Result<DynStreamingIterator<'a, CompressedPage, E>, E>>,
    ordinal: usize,
) -> ParquetResult<(RowGroup, Vec<Vec<PageWriteSpec>>, u64)>
where
    W: AsyncWrite + Unpin + Send,
    ParquetError: From<E>,
    E: std::error::Error,
{
    let column_iter = descriptors.iter().zip(columns);

    let initial = offset;
    let mut columns = vec![];
    for (descriptor, page_iter) in column_iter {
        let (column, page_specs, size) =
            write_column_chunk_async(writer, offset, descriptor, page_iter?).await?;
        offset += size;
        columns.push((column, page_specs));
    }
    let bytes_written = offset - initial;

    let num_rows = compute_num_rows(&columns)?;

    // compute row group stats
    let file_offset = columns
        .first()
        .map(|(column_chunk, _)| {
            ColumnOffsetsMetadata::from_column_chunk(column_chunk).calc_row_group_file_offset()
        })
        .unwrap_or(None);

    let total_byte_size = columns
        .iter()
        .map(|(c, _)| c.meta_data.as_ref().unwrap().total_uncompressed_size)
        .sum();
    let total_compressed_size = columns
        .iter()
        .map(|(c, _)| c.meta_data.as_ref().unwrap().total_compressed_size)
        .sum();

    let (columns, specs) = columns.into_iter().unzip();

    Ok((
        RowGroup {
            columns,
            total_byte_size,
            num_rows: num_rows as i64,
            sorting_columns: None,
            file_offset,
            total_compressed_size: Some(total_compressed_size),
            ordinal: ordinal.try_into().ok(),
        },
        specs,
        bytes_written,
    ))
}
