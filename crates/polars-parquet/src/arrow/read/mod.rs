//! APIs to read from Parquet format.
#![allow(clippy::type_complexity)]

mod deserialize;
pub mod schema;
pub mod statistics;

use std::io::{Read, Seek};

use arrow::types::{i256, NativeType};
pub use deserialize::{
    column_iter_to_arrays, create_list, create_map, get_page_iterator, init_nested, n_columns,
    Filter, InitNested, NestedState,
};
#[cfg(feature = "async")]
use futures::{AsyncRead, AsyncSeek};
use polars_error::PolarsResult;
pub use schema::{infer_schema, FileMetaData};

use crate::parquet::error::ParquetResult;
#[cfg(feature = "async")]
pub use crate::parquet::read::{get_page_stream, read_metadata_async as _read_metadata_async};
// re-exports of crate::parquet's relevant APIs
pub use crate::parquet::{
    error::ParquetError,
    fallible_streaming_iterator,
    metadata::{ColumnChunkMetaData, ColumnDescriptor, RowGroupMetaData},
    page::{CompressedDataPage, DataPageHeader, Page},
    read::{
        decompress, get_column_iterator, read_metadata as _read_metadata, BasicDecompressor,
        MutStreamingIterator, PageReader, ReadColumnIterator, State,
    },
    schema::types::{
        GroupLogicalType, ParquetType, PhysicalType, PrimitiveConvertedType, PrimitiveLogicalType,
        TimeUnit as ParquetTimeUnit,
    },
    types::int96_to_i64_ns,
    FallibleStreamingIterator,
};

/// Returns all [`ColumnChunkMetaData`] associated to `field_name`.
/// For non-nested parquet types, this returns a single column
pub fn get_field_pages<'a, T>(
    columns: &'a [ColumnChunkMetaData],
    items: &'a [T],
    field_name: &str,
) -> Vec<&'a T> {
    columns
        .iter()
        .zip(items)
        .filter(|(metadata, _)| metadata.descriptor().path_in_schema[0] == field_name)
        .map(|(_, item)| item)
        .collect()
}

/// Reads parquets' metadata synchronously.
pub fn read_metadata<R: Read + Seek>(reader: &mut R) -> PolarsResult<FileMetaData> {
    Ok(_read_metadata(reader)?)
}

/// Reads parquets' metadata asynchronously.
#[cfg(feature = "async")]
pub async fn read_metadata_async<R: AsyncRead + AsyncSeek + Send + Unpin>(
    reader: &mut R,
) -> PolarsResult<FileMetaData> {
    Ok(_read_metadata_async(reader).await?)
}

fn convert_days_ms(value: &[u8]) -> arrow::types::days_ms {
    arrow::types::days_ms(
        i32::from_le_bytes(value[4..8].try_into().unwrap()),
        i32::from_le_bytes(value[8..12].try_into().unwrap()),
    )
}

fn convert_i128(value: &[u8], n: usize) -> i128 {
    // Copy the fixed-size byte value to the start of a 16 byte stack
    // allocated buffer, then use an arithmetic right shift to fill in
    // MSBs, which accounts for leading 1's in negative (two's complement)
    // values.
    let mut bytes = [0u8; 16];
    bytes[..n].copy_from_slice(value);
    i128::from_be_bytes(bytes) >> (8 * (16 - n))
}

fn convert_i256(value: &[u8]) -> i256 {
    if value[0] >= 128 {
        let mut neg_bytes = [255u8; 32];
        neg_bytes[32 - value.len()..].copy_from_slice(value);
        i256::from_be_bytes(neg_bytes)
    } else {
        let mut bytes = [0u8; 32];
        bytes[32 - value.len()..].copy_from_slice(value);
        i256::from_be_bytes(bytes)
    }
}
