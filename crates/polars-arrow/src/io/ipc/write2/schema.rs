use polars_buffer::Buffer;

use crate::io::ipc::write2::message::serialize_ipc_flatbuf;

/// Converts a [ArrowSchema] and [IpcField]s to a flatbuffers-encoded [polars_arrow_format::ipc::Message].
pub fn serialize_ipc_schema_message_bytes(
    serialized_ipc_schema: Box<polars_arrow_format::ipc::Schema>,
) -> Buffer<u8> {
    let message = polars_arrow_format::ipc::Message {
        version: polars_arrow_format::ipc::MetadataVersion::V5,
        header: Some(polars_arrow_format::ipc::MessageHeader::Schema(
            serialized_ipc_schema,
        )),
        body_length: 0,
        custom_metadata: None,
    };

    serialize_ipc_flatbuf(message)
}
