use std::io::Cursor;

use polars_buffer::Buffer;
use polars_parquet_format::thrift::protocol::TCompactInputProtocol;
use polars_parquet_format::{
    ColumnChunk, ColumnMetaData, FileMetaData as ThriftFileMetaData, RowGroup,
    Statistics as ThriftStatistics,
};

use super::compact::{
    ByteRange, CompactColumnChunk, CompactColumnMetaData, CompactFileMetaData, CompactRowGroup,
    CompactStatistics,
};
use crate::parquet::compression::Compression;
use crate::parquet::encryption::decrypt::{CryptoContext, FileDecryptor};
use crate::parquet::error::{ParquetError, ParquetResult};

pub(crate) fn decode_thrift_file_metadata(
    footer: &[u8],
    file_decryptor: Option<FileDecryptor>,
) -> ParquetResult<CompactFileMetaData> {
    let mut cursor = Cursor::new(footer);
    let mut protocol = TCompactInputProtocol::new(&mut cursor, usize::MAX);
    let thrift = ThriftFileMetaData::read_from_in_protocol(&mut protocol)?;
    thrift_file_metadata_to_compact(thrift, file_decryptor)
}

fn thrift_file_metadata_to_compact(
    thrift: ThriftFileMetaData,
    file_decryptor: Option<FileDecryptor>,
) -> ParquetResult<CompactFileMetaData> {
    let mut footer = vec![];
    let row_groups = thrift
        .row_groups
        .into_iter()
        .enumerate()
        .map(|(row_group_index, row_group)| {
            row_group_to_compact(
                row_group,
                row_group_index,
                file_decryptor.as_ref(),
                &mut footer,
            )
        })
        .collect::<ParquetResult<Vec<_>>>()?;

    Ok(CompactFileMetaData {
        version: thrift.version,
        schema: thrift.schema,
        num_rows: thrift.num_rows,
        row_groups,
        key_value_metadata: thrift.key_value_metadata,
        created_by: thrift.created_by,
        column_orders: thrift.column_orders,
        file_decryptor,
        footer_buf: Buffer::from_vec(footer),
    })
}

fn row_group_to_compact(
    row_group: RowGroup,
    row_group_index: usize,
    file_decryptor: Option<&FileDecryptor>,
    footer: &mut Vec<u8>,
) -> ParquetResult<CompactRowGroup> {
    let columns = row_group
        .columns
        .into_iter()
        .enumerate()
        .map(|(column_index, column_chunk)| {
            column_chunk_to_compact(
                column_chunk,
                row_group_index,
                column_index,
                file_decryptor,
                footer,
            )
        })
        .collect::<ParquetResult<Vec<_>>>()?;

    Ok(CompactRowGroup {
        columns,
        total_byte_size: row_group.total_byte_size,
        num_rows: row_group.num_rows,
        sorting_columns: row_group.sorting_columns,
    })
}

fn column_chunk_to_compact(
    mut column_chunk: ColumnChunk,
    row_group_index: usize,
    column_index: usize,
    file_decryptor: Option<&FileDecryptor>,
    footer: &mut Vec<u8>,
) -> ParquetResult<CompactColumnChunk> {
    let meta_data = if file_decryptor.is_some() && column_chunk.encrypted_column_metadata.is_some()
    {
        decrypt_column_metadata(
            &mut column_chunk,
            row_group_index,
            column_index,
            file_decryptor,
        )?
    } else {
        column_chunk
            .meta_data
            .take()
            .ok_or_else(|| ParquetError::oos("ColumnChunk.meta_data missing"))?
    };

    Ok(CompactColumnChunk {
        meta_data: column_metadata_to_compact(meta_data, footer)?,
        offset_index_offset: column_chunk.offset_index_offset,
        offset_index_length: column_chunk.offset_index_length,
        column_index_offset: column_chunk.column_index_offset,
        column_index_length: column_chunk.column_index_length,
        crypto_metadata: column_chunk.crypto_metadata,
    })
}

fn decrypt_column_metadata(
    column_chunk: &mut ColumnChunk,
    row_group_index: usize,
    column_index: usize,
    file_decryptor: Option<&FileDecryptor>,
) -> ParquetResult<ColumnMetaData> {
    let file_decryptor = file_decryptor.ok_or_else(|| {
        ParquetError::oos("column metadata is encrypted but no file decryptor is available")
    })?;
    let crypto_metadata = column_chunk
        .crypto_metadata
        .as_ref()
        .ok_or_else(|| ParquetError::oos("encrypted column metadata is missing crypto_metadata"))?;
    let encrypted_column_metadata = column_chunk
        .encrypted_column_metadata
        .take()
        .ok_or_else(|| ParquetError::oos("ColumnChunk.meta_data missing"))?;

    let crypto_context = CryptoContext::for_column(
        file_decryptor,
        crypto_metadata,
        row_group_index,
        column_index,
    )?;
    let aad = crypto_context.create_column_metadata_aad()?;
    let decrypted = crypto_context
        .metadata_decryptor()
        .decrypt(&encrypted_column_metadata, &aad)
        .map_err(|_| ParquetError::oos("failed to decrypt column metadata"))?;

    let mut cursor = Cursor::new(decrypted.as_slice());
    let mut protocol = TCompactInputProtocol::new(&mut cursor, usize::MAX);
    Ok(ColumnMetaData::read_from_in_protocol(&mut protocol)?)
}

fn column_metadata_to_compact(
    meta_data: ColumnMetaData,
    footer: &mut Vec<u8>,
) -> ParquetResult<CompactColumnMetaData> {
    let codec = Compression::try_from(meta_data.codec)?;
    Ok(CompactColumnMetaData {
        codec,
        num_values: meta_data.num_values,
        total_uncompressed_size: meta_data.total_uncompressed_size,
        total_compressed_size: meta_data.total_compressed_size,
        data_page_offset: meta_data.data_page_offset,
        index_page_offset: meta_data.index_page_offset,
        dictionary_page_offset: meta_data.dictionary_page_offset,
        statistics: meta_data
            .statistics
            .map(|statistics| statistics_to_compact(statistics, footer)),
        bloom_filter_offset: meta_data.bloom_filter_offset,
        bloom_filter_length: meta_data.bloom_filter_length,
    })
}

fn statistics_to_compact(statistics: ThriftStatistics, footer: &mut Vec<u8>) -> CompactStatistics {
    CompactStatistics {
        null_count: statistics.null_count,
        distinct_count: statistics.distinct_count,
        max_value: statistics
            .max_value
            .as_deref()
            .map(|bytes| append_to_footer(footer, bytes)),
        min_value: statistics
            .min_value
            .as_deref()
            .map(|bytes| append_to_footer(footer, bytes)),
        is_max_value_exact: statistics.is_max_value_exact,
        is_min_value_exact: statistics.is_min_value_exact,
    }
}

fn append_to_footer(footer: &mut Vec<u8>, bytes: &[u8]) -> ByteRange {
    let offset = footer.len() as u32;
    footer.extend_from_slice(bytes);
    ByteRange {
        offset,
        len: bytes.len() as u32,
    }
}
