use std::io::Write;

use bytes::Bytes;
use polars_error::PolarsResult;

use super::super::CONTINUATION_MARKER;
use super::common::{EncodedData, pad_to_64};
use crate::io::ipc::write::{EncodedDataBytes, PutOwned};

/// Write a message's IPC data and buffers, returning metadata and buffer data lengths written
pub fn write_message<W: Write>(
    writer: &mut W,
    encoded: &EncodedData,
) -> PolarsResult<(usize, usize)> {
    let arrow_data_len = encoded.arrow_data.len();

    let a = 8 - 1;
    let buffer = &encoded.ipc_message;
    let flatbuf_size = buffer.len();
    let prefix_size = 8;
    let aligned_size = (flatbuf_size + prefix_size + a) & !a;
    let padding_bytes = aligned_size - flatbuf_size - prefix_size;

    write_continuation(writer, (aligned_size - prefix_size) as i32)?;

    // write the flatbuf
    if flatbuf_size > 0 {
        writer.write_all(buffer)?;
    }
    // write padding
    // aligned to a 8 byte boundary, so maximum is [u8;8]
    const PADDING_MAX: [u8; 8] = [0u8; 8];
    writer.write_all(&PADDING_MAX[..padding_bytes])?;

    // write arrow data
    let body_len = if arrow_data_len > 0 {
        write_body_buffers(writer, &encoded.arrow_data)?
    } else {
        0
    };

    Ok((aligned_size, body_len))
}

/// Put a message's IPC data and buffers, returning metadata and buffer data lengths written.
/// Unlike `write_message`, the encoded data is consumed.
pub fn put_message<W: Write + PutOwned>(
    writer: &mut W,
    encoded: EncodedDataBytes,
) -> PolarsResult<(usize, usize)> {
    let arrow_data_len = encoded.arrow_data.len();

    let a = 8 - 1;
    let buffer = &encoded.ipc_message;
    let flatbuf_size = buffer.len();
    let prefix_size = 8;
    let aligned_size = (flatbuf_size + prefix_size + a) & !a;
    let padding_bytes = aligned_size - flatbuf_size - prefix_size;

    put_continuation(writer, (aligned_size - prefix_size) as i32)?;

    // write the flatbuf
    if flatbuf_size > 0 {
        writer.put(encoded.ipc_message)?;
    }
    // write padding
    // aligned to a 8 byte boundary, so maximum is [u8;8]
    const PADDING_MAX: [u8; 8] = [0u8; 8];
    writer.put(Bytes::from(&PADDING_MAX[..padding_bytes]))?;

    // write arrow data
    let body_len = if arrow_data_len > 0 {
        put_body_buffers(writer, encoded.arrow_data)?
    } else {
        0
    };

    Ok((aligned_size, body_len))
}

fn write_body_buffers<W: Write>(mut writer: W, data: &[u8]) -> PolarsResult<usize> {
    let len = data.len();
    let pad_len = pad_to_64(data.len());
    let total_len = len + pad_len;

    // write body buffer
    writer.write_all(data)?;
    if pad_len > 0 {
        writer.write_all(&vec![0u8; pad_len][..])?;
    }

    Ok(total_len)
}

fn put_body_buffers<W: Write + PutOwned>(mut writer: W, data: Bytes) -> PolarsResult<usize> {
    let len = data.len();
    let pad_len = pad_to_64(data.len());
    let total_len = len + pad_len;

    // write body buffer
    writer.put(data)?;
    // writer.write_all(data)?;

    if pad_len > 0 {
        writer.put(Bytes::from(vec![0u8; pad_len]))?;
        // writer.write_all(&vec![0u8; pad_len][..])?;
    }

    Ok(total_len)
}

/// Write a record batch to the writer, writing the message size before the message
/// if the record batch is being written to a stream
pub fn write_continuation<W: Write>(writer: &mut W, total_len: i32) -> PolarsResult<usize> {
    writer.write_all(&CONTINUATION_MARKER)?;
    writer.write_all(&total_len.to_le_bytes()[..])?;
    Ok(8)
}

pub fn put_continuation<W: Write + PutOwned>(
    writer: &mut W,
    total_len: i32,
) -> PolarsResult<usize> {
    let mut buf = [0u8; 8];
    buf[..4].copy_from_slice(&CONTINUATION_MARKER);
    buf[4..].copy_from_slice(&total_len.to_le_bytes());
    writer.put(Bytes::copy_from_slice(&buf))?;
    Ok(8)
}
