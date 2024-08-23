use futures::future::BoxFuture;
use futures::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt};

use crate::parquet::error::ParquetError;
use crate::parquet::metadata::ColumnChunkMetaData;

/// Reads a single column chunk into memory asynchronously
pub async fn read_column_async<'b, R, F>(
    factory: F,
    meta: &ColumnChunkMetaData,
) -> Result<Vec<u8>, ParquetError>
where
    R: AsyncRead + AsyncSeek + Send + Unpin,
    F: Fn() -> BoxFuture<'b, std::io::Result<R>>,
{
    let mut reader = factory().await?;
    let (start, length) = meta.byte_range();
    reader.seek(std::io::SeekFrom::Start(start)).await?;

    let mut chunk = vec![];
    chunk.try_reserve(length as usize)?;
    reader.take(length).read_to_end(&mut chunk).await?;
    Result::Ok(chunk)
}
