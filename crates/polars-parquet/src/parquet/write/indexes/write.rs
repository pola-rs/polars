use futures::AsyncWrite;
use polars_parquet_format::thrift::protocol::TCompactOutputStreamProtocol;

use super::serialize::{serialize_column_index, serialize_offset_index};
use crate::parquet::error::ParquetResult;
use crate::parquet::write::page::PageWriteSpec;

/// Returns the number of bytes written, or `None` if the pages have no valid column index.
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn write_column_index_async<W: AsyncWrite + Unpin + Send>(
    writer: &mut W,
    pages: &[PageWriteSpec],
) -> ParquetResult<Option<u64>> {
    let Some(index) = serialize_column_index(pages)? else {
        return Ok(None);
    };
    let mut protocol = TCompactOutputStreamProtocol::new(writer);
    Ok(Some(
        index.write_to_out_stream_protocol(&mut protocol).await? as u64,
    ))
}

#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn write_offset_index_async<W: AsyncWrite + Unpin + Send>(
    writer: &mut W,
    pages: &[PageWriteSpec],
) -> ParquetResult<u64> {
    let index = serialize_offset_index(pages)?;
    let mut protocol = TCompactOutputStreamProtocol::new(&mut *writer);
    Ok(index.write_to_out_stream_protocol(&mut protocol).await? as u64)
}
