use futures::future::{try_join_all, BoxFuture};
use futures::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt};

use crate::error::Error;
use crate::metadata::ColumnChunkMetaData;
use crate::read::get_field_columns;

/// Reads a single column chunk into memory asynchronously
pub async fn read_column_async<'b, R, F>(
    factory: F,
    meta: &ColumnChunkMetaData,
) -> Result<Vec<u8>, Error>
where
    R: AsyncRead + AsyncSeek + Send + Unpin,
    F: Fn() -> BoxFuture<'b, std::io::Result<R>>,
{
    let mut reader = factory().await?;
    let (start, length) = meta.byte_range();
    reader.seek(std::io::SeekFrom::Start(start)).await?;

    let mut chunk = vec![];
    chunk.try_reserve(length as usize)?;
    reader.take(length as u64).read_to_end(&mut chunk).await?;
    Result::Ok(chunk)
}

/// Reads all columns that are part of the parquet field `field_name`
/// # Implementation
/// This operation is IO-bounded `O(C)` where C is the number of columns associated to
/// the field (one for non-nested types)
///
/// It does so asynchronously via a single `join_all` over all the necessary columns for
/// `field_name`.
pub async fn read_columns_async<
    'a,
    'b,
    R: AsyncRead + AsyncSeek + Send + Unpin,
    F: Fn() -> BoxFuture<'b, std::io::Result<R>> + Clone,
>(
    factory: F,
    columns: &'a [ColumnChunkMetaData],
    field_name: &'a str,
) -> Result<Vec<(&'a ColumnChunkMetaData, Vec<u8>)>, Error> {
    let fields = get_field_columns(columns, field_name).collect::<Vec<_>>();
    let futures = fields
        .iter()
        .map(|meta| async { read_column_async(factory.clone(), meta).await });

    let columns = try_join_all(futures).await?;
    Ok(fields.into_iter().zip(columns).collect())
}
