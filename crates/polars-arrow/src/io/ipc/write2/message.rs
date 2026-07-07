use arrow_format::ipc::{KeyValue, planus};
use polars_buffer::Buffer;
use polars_error::PolarsResult;

use crate::io::ipc::CONTINUATION_MARKER;
use crate::io::ipc::write::common::serialize_compression;
use crate::io::ipc::write2::array::IpcBatchSerializationContext;
use crate::io::write_owned::WriteBytesOwned;

/// # Returns
/// Buffer containing continuation marker and length.
pub fn finish_encode_ipc_record_batch(
    ctx: &mut IpcBatchSerializationContext<'_>,
    num_rows: usize,
    custom_metadata: Option<Vec<KeyValue>>,
) -> PolarsResult<Vec<u8>> {
    let compression = serialize_compression(ctx.compression);
    let variadic_buffer_counts = (!ctx.variadic_buffer_counts.is_empty())
        .then(|| std::mem::take(&mut ctx.variadic_buffer_counts));

    let message = arrow_format::ipc::Message {
        version: arrow_format::ipc::MetadataVersion::V5,
        header: Some(arrow_format::ipc::MessageHeader::RecordBatch(Box::new(
            arrow_format::ipc::RecordBatch {
                length: num_rows as i64,
                nodes: Some(std::mem::take(&mut ctx.field_nodes)),
                buffers: Some(std::mem::take(&mut ctx.buffers)),
                compression,
                variadic_buffer_counts,
            },
        ))),
        body_length: ctx.arrow_data.len() as i64,
        custom_metadata,
    };

    ctx.ipc_message
        .write_all_owned(&serialize_ipc_flatbuf(message))?;
    let continuation = finish_ipc_message_bytes(ctx.ipc_message)?;

    if let Some(padding) = ctx
        .arrow_data
        .len()
        .checked_next_multiple_of(64)
        .map(|x| x - ctx.arrow_data.len())
        && padding != 0
    {
        ctx.arrow_data.write_all_owned(&Buffer::zeroed(padding))?;
    }

    Ok(continuation)
}

/// # Returns
/// Buffer containing continuation marker and length.
pub fn finish_encode_ipc_dictionary_batch(
    ctx: &mut IpcBatchSerializationContext<'_>,
    num_rows: usize,
    dictionary_id: i64,
) -> PolarsResult<Vec<u8>> {
    let compression = serialize_compression(ctx.compression);
    let variadic_buffer_counts = (!ctx.variadic_buffer_counts.is_empty())
        .then(|| std::mem::take(&mut ctx.variadic_buffer_counts));

    let message = arrow_format::ipc::Message {
        version: arrow_format::ipc::MetadataVersion::V5,
        header: Some(arrow_format::ipc::MessageHeader::DictionaryBatch(Box::new(
            arrow_format::ipc::DictionaryBatch {
                id: dictionary_id,
                data: Some(Box::new(arrow_format::ipc::RecordBatch {
                    length: num_rows as i64,
                    nodes: Some(std::mem::take(&mut ctx.field_nodes)),
                    buffers: Some(std::mem::take(&mut ctx.buffers)),
                    compression,
                    variadic_buffer_counts,
                })),
                is_delta: false,
            },
        ))),
        body_length: ctx.arrow_data.len() as i64,
        custom_metadata: None,
    };

    ctx.ipc_message
        .write_all_owned(&serialize_ipc_flatbuf(message))?;
    let continuation = finish_ipc_message_bytes(ctx.ipc_message)?;

    if let Some(padding) = ctx
        .arrow_data
        .len()
        .checked_next_multiple_of(64)
        .map(|x| x - ctx.arrow_data.len())
        && padding != 0
    {
        ctx.arrow_data.write_all_owned(&Buffer::zeroed(padding))?;
    }

    Ok(continuation)
}

pub fn serialize_ipc_flatbuf<T>(flatbuf: impl planus::WriteAsOffset<T>) -> Buffer<u8> {
    let mut builder = planus::Builder::new();
    builder.finish(&flatbuf, None);

    return Buffer::from_owner(AsSliceWrap(builder));

    struct AsSliceWrap(planus::Builder);

    impl AsRef<[u8]> for AsSliceWrap {
        fn as_ref(&self) -> &[u8] {
            self.0.as_slice()
        }
    }
}

/// Pads the `ipc_message`.
///
/// # Returns
/// Buffer containing continuation marker and length.
pub fn finish_ipc_message_bytes(ipc_message: &mut dyn WriteBytesOwned) -> PolarsResult<Vec<u8>> {
    let prefix_len: usize = 8;

    let aligned_size = {
        let size = prefix_len + ipc_message.len();
        let padding = size.checked_next_multiple_of(8).map_or(0, |x| x - size);

        if padding != 0 {
            ipc_message.write_all_owned(&Buffer::zeroed(padding))?;
        }

        size + padding
    };

    let mut ret: Vec<u8> = Vec::with_capacity(prefix_len);
    ret.extend_from_slice(&CONTINUATION_MARKER);
    ret.extend_from_slice(&i32::to_le_bytes((aligned_size - prefix_len) as _));

    Ok(ret)
}
