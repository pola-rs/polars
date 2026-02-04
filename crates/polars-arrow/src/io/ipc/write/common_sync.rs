use std::io::Write;
use std::sync::Arc;

use arrow_format::ipc::planus::Builder;
use bytes::Bytes;
use polars_error::PolarsResult;

use super::super::{ARROW_MAGIC_V2, ARROW_MAGIC_V2_PADDED, CONTINUATION_MARKER};
use super::common::{EncodedData, pad_to_64};
use super::schema;
use crate::datatypes::*;
use crate::io::ipc::IpcField;
use crate::io::ipc::write::EncodedDataBytes;

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

/// Encapsulate an encoded IPC message into the provided Bytes queue. Ownership
/// of the data will move. Returns the metadata and body length in bytes.
pub fn push_message(queue: &mut Vec<Bytes>, encoded: EncodedDataBytes) -> (usize, usize) {
    let arrow_data_len = encoded.arrow_data.len();

    let a = 8 - 1;
    let buffer = encoded.ipc_message;
    let flatbuf_size = buffer.len();
    let prefix_size = 8;
    let aligned_size = (flatbuf_size + prefix_size + a) & !a;
    let padding_bytes = aligned_size - flatbuf_size - prefix_size;

    // Continuation.
    let total_len = (aligned_size - prefix_size) as i32;
    queue.push(Bytes::from_static(&CONTINUATION_MARKER));
    queue.push(Bytes::copy_from_slice(&total_len.to_le_bytes()[..]));

    // Write the flatbuf.
    if flatbuf_size > 0 {
        queue.push(buffer);
    }

    // Write padding.
    // This is aligned to a 8 byte boundary, so maximum is [u8;8].
    const PADDING_MAX: [u8; 8] = [0u8; 8];
    queue.push(Bytes::from_static(&PADDING_MAX[..padding_bytes]));

    // write arrow data
    let body_len = if arrow_data_len > 0 {
        let data = encoded.arrow_data;
        let len = data.len();
        let pad_len = pad_to_64(data.len());
        let total_len = len + pad_len;

        queue.push(data);
        if pad_len > 0 {
            queue.push(Bytes::from(vec![0u8; pad_len]));
        }
        total_len
    } else {
        0
    };

    (aligned_size, body_len)
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

/// Write a record batch to the writer, writing the message size before the message
/// if the record batch is being written to a stream
pub fn write_continuation<W: Write>(writer: &mut W, total_len: i32) -> PolarsResult<usize> {
    writer.write_all(&CONTINUATION_MARKER)?;
    writer.write_all(&total_len.to_le_bytes()[..])?;
    Ok(8)
}

/// Push the IPC magic bytes.
pub fn push_magic(queue: &mut Vec<Bytes>, padded: bool) -> usize {
    if padded {
        queue.push(Bytes::from_static(&ARROW_MAGIC_V2_PADDED));
        8
    } else {
        queue.push(Bytes::from_static(&ARROW_MAGIC_V2));
        6
    }
}

/// Append a continuation marker and the `total_len` value to the Bytes queue.
pub fn push_continuation(queue: &mut Vec<Bytes>, total_len: i32) -> usize {
    let mut buf = [0u8; 8];
    buf[..4].copy_from_slice(&CONTINUATION_MARKER);
    buf[4..].copy_from_slice(&total_len.to_le_bytes());
    queue.push(Bytes::copy_from_slice(&buf));
    8
}

/// Build the IPC File Footer and accumulate the owned Bytes into the queue.
/// Returns the total number of bytes added.
pub fn push_footer(
    queue: &mut Vec<Bytes>,
    schema: &ArrowSchema,
    ipc_fields: &[IpcField],
    dictionary_blocks: Vec<arrow_format::ipc::Block>,
    record_blocks: Vec<arrow_format::ipc::Block>,
    // Placeholder, inherited from FileWriter for future use.
    custom_schema_metadata: Option<Arc<Metadata>>,
) -> usize {
    let mut total_len = 0;

    total_len += push_continuation(queue, 0);

    let schema = schema::serialize_schema(schema, ipc_fields, custom_schema_metadata.as_deref());

    let root = arrow_format::ipc::Footer {
        version: arrow_format::ipc::MetadataVersion::V5,
        schema: Some(Box::new(schema)),
        dictionaries: Some(dictionary_blocks),
        record_batches: Some(record_blocks),
        custom_metadata: None,
    };
    let mut builder = Builder::new();
    let footer_data = builder.finish(&root, None);
    let footer_data = Bytes::copy_from_slice(footer_data);
    let footer_data_len = footer_data.len();

    queue.push(footer_data);
    total_len += footer_data_len;

    queue.push(Bytes::copy_from_slice(
        &(footer_data_len as i32).to_le_bytes(),
    ));
    total_len += 4;

    total_len
}
