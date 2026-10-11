//! APIs to read from Parquet format.
#![allow(clippy::type_complexity)]

mod deserialize;
pub mod expr;
pub mod schema;
pub mod statistics;

use std::io::{Read, Seek};

pub(crate) use deserialize::convert_i128;
pub use deserialize::{
    Filter, InitNested, NestedState, PredicateFilter, column_iter_to_arrays, create_list,
    create_map, init_nested, n_columns,
};
#[cfg(feature = "async")]
use futures::{AsyncRead, AsyncSeek};
use polars_arrow::types::{NativeType, i256};
use polars_error::PolarsResult;
pub use schema::{FileMetadata, infer_schema};

#[cfg(feature = "async")]
pub use crate::parquet::read::read_metadata_async as _read_metadata_async;
// re-exports of crate::parquet's relevant APIs
pub use crate::parquet::{
    FallibleStreamingIterator,
    error::ParquetError,
    fallible_streaming_iterator,
    metadata::{ColumnChunkMetadata, ColumnDescriptor, RowGroupMetadata},
    page::{CompressedDataPage, DataPageHeader, Page},
    read::{
        BasicDecompressor, MutStreamingIterator, PageReader, ReadColumnIterator, State, decompress,
        get_column_iterator, read_metadata as _read_metadata,
    },
    schema::types::{
        GroupLogicalType, ParquetType, PhysicalType, PrimitiveConvertedType, PrimitiveLogicalType,
        TimeUnit as ParquetTimeUnit,
    },
    types::int96_to_i64_ns,
};

/// Reads parquets' metadata synchronously.
pub fn read_metadata<R: Read + Seek>(reader: &mut R) -> PolarsResult<FileMetadata> {
    Ok(_read_metadata(reader)?)
}

/// Reads parquets' metadata asynchronously.
#[cfg(feature = "async")]
pub async fn read_metadata_async<R: AsyncRead + AsyncSeek + Send + Unpin>(
    reader: &mut R,
) -> PolarsResult<FileMetadata> {
    Ok(_read_metadata_async(reader).await?)
}

fn convert_days_ms(value: &[u8]) -> polars_arrow::types::days_ms {
    polars_arrow::types::days_ms(
        i32::from_le_bytes(value[4..8].try_into().unwrap()),
        i32::from_le_bytes(value[8..12].try_into().unwrap()),
    )
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
