use futures::{AsyncWrite, AsyncWriteExt};
use polars_error::PolarsResult;

use super::super::CONTINUATION_MARKER;
use super::common::{pad_to_64, EncodedData};

/// Write a message's IPC data and buffers, returning metadata and buffer data lengths written
pub async fn write_message<W: AsyncWrite + Unpin + Send>(
    mut writer: W,
    encoded: EncodedData,
) -> PolarsResult<(usize, usize)> {
    let arrow_data_len = encoded.arrow_data.len();

    let a = 64 - 1;
    let buffer = encoded.ipc_message;
    let flatbuf_size = buffer.len();
    let prefix_size = 8; // the message length
    let aligned_size = (flatbuf_size + prefix_size + a) & !a;
    let padding_bytes = aligned_size - flatbuf_size - prefix_size;

    write_continuation(&mut writer, (aligned_size - prefix_size) as i32).await?;

    // write the flatbuf
    if flatbuf_size > 0 {
        writer.write_all(&buffer).await?;
    }
    // write padding
    writer.write_all(&vec![0; padding_bytes]).await?;

    // write arrow data
    let body_len = if arrow_data_len > 0 {
        write_body_buffers(writer, &encoded.arrow_data).await?
    } else {
        0
    };

    Ok((aligned_size, body_len))
}

/// Write a record batch to the writer, writing the message size before the message
/// if the record batch is being written to a stream
pub async fn write_continuation<W: AsyncWrite + Unpin + Send>(
    mut writer: W,
    total_len: i32,
) -> PolarsResult<usize> {
    writer.write_all(&CONTINUATION_MARKER).await?;
    writer.write_all(&total_len.to_le_bytes()[..]).await?;
    Ok(8)
}

async fn write_body_buffers<W: AsyncWrite + Unpin + Send>(
    mut writer: W,
    data: &[u8],
) -> PolarsResult<usize> {
    let len = data.len();
    let pad_len = pad_to_64(data.len());
    let total_len = len + pad_len;

    // write body buffer
    writer.write_all(data).await?;
    if pad_len > 0 {
        writer.write_all(&vec![0u8; pad_len][..]).await?;
    }

    Ok(total_len)
}
