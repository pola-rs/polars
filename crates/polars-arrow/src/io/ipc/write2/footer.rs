use arrow_format::ipc::KeyValue;
use polars_buffer::Buffer;
use polars_error::PolarsResult;

use crate::io::ipc::write2::message::serialize_ipc_flatbuf;
use crate::io::ipc::{ARROW_MAGIC_V2, CONTINUATION_MARKER};
use crate::io::write_owned::WriteBytesOwned;

/// Serialize the IPC footer into `writer`, ending with the arrow magic for the
/// end of the IPC file.
pub fn serialize_ipc_footer_and_magic_bytes(
    writer: &mut dyn WriteBytesOwned,
    serialized_ipc_schema: Box<arrow_format::ipc::Schema>,
    dictionary_blocks: Vec<arrow_format::ipc::Block>,
    record_blocks: Vec<arrow_format::ipc::Block>,
    custom_metadata: Option<Vec<(String, String)>>,
) -> PolarsResult<()> {
    // Note: Length at 4..8 is 0.
    let mut prefix: Vec<u8> = vec![0; 8];
    prefix[..4].copy_from_slice(&CONTINUATION_MARKER);

    writer.write_all_owned(&Buffer::from_vec(prefix))?;

    let footer = arrow_format::ipc::Footer {
        version: arrow_format::ipc::MetadataVersion::V5,
        schema: Some(serialized_ipc_schema),
        dictionaries: Some(dictionary_blocks),
        record_batches: Some(record_blocks),
        custom_metadata: custom_metadata.map(|kv_vec| {
            kv_vec
                .into_iter()
                .map(|(k, v)| KeyValue {
                    key: Some(k),
                    value: Some(v),
                })
                .collect()
        }),
    };

    let footer_bytes = serialize_ipc_flatbuf(footer);
    let footer_bytes_len = footer_bytes.len();
    writer.write_all_owned(&footer_bytes)?;
    writer.write_all(&(footer_bytes_len as i32).to_le_bytes())?;
    writer.write_all_owned(&Buffer::from_static(&ARROW_MAGIC_V2))?;

    Ok(())
}
