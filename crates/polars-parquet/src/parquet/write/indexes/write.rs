use std::io::Write;

#[cfg(feature = "async")]
use futures::AsyncWrite;
#[cfg(feature = "async")]
use parquet_format_safe::thrift::protocol::TCompactOutputStreamProtocol;

use parquet_format_safe::thrift::protocol::TCompactOutputProtocol;

use crate::error::Result;
pub use crate::metadata::KeyValue;

use crate::write::page::PageWriteSpec;

use super::serialize::{serialize_column_index, serialize_offset_index};

pub fn write_column_index<W: Write>(writer: &mut W, pages: &[PageWriteSpec]) -> Result<u64> {
    let index = serialize_column_index(pages)?;
    let mut protocol = TCompactOutputProtocol::new(writer);
    Ok(index.write_to_out_protocol(&mut protocol)? as u64)
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn write_column_index_async<W: AsyncWrite + Unpin + Send>(
    writer: &mut W,
    pages: &[PageWriteSpec],
) -> Result<u64> {
    let index = serialize_column_index(pages)?;
    let mut protocol = TCompactOutputStreamProtocol::new(writer);
    Ok(index.write_to_out_stream_protocol(&mut protocol).await? as u64)
}

pub fn write_offset_index<W: Write>(writer: &mut W, pages: &[PageWriteSpec]) -> Result<u64> {
    let index = serialize_offset_index(pages)?;
    let mut protocol = TCompactOutputProtocol::new(&mut *writer);
    Ok(index.write_to_out_protocol(&mut protocol)? as u64)
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn write_offset_index_async<W: AsyncWrite + Unpin + Send>(
    writer: &mut W,
    pages: &[PageWriteSpec],
) -> Result<u64> {
    let index = serialize_offset_index(pages)?;
    let mut protocol = TCompactOutputStreamProtocol::new(&mut *writer);
    Ok(index.write_to_out_stream_protocol(&mut protocol).await? as u64)
}
