use std::fmt::{Display, Formatter};

/// The different types of errors that reading from IPC can cause
#[derive(Debug)]
#[non_exhaustive]
pub enum OutOfSpecKind {
    /// The IPC file does not start with [b'A', b'R', b'R', b'O', b'W', b'1']
    InvalidHeader,
    /// The IPC file does not end with [b'A', b'R', b'R', b'O', b'W', b'1']
    InvalidFooter,
    /// The first 4 bytes of the last 10 bytes is < 0
    NegativeFooterLength,
    /// The footer is an invalid flatbuffer
    InvalidFlatbufferFooter(polars_arrow_format::ipc::planus::Error),
    /// The file's footer does not contain record batches
    MissingRecordBatches,
    /// The footer's record batches is an invalid flatbuffer
    InvalidFlatbufferRecordBatches(polars_arrow_format::ipc::planus::Error),
    /// The file's footer does not contain a schema
    MissingSchema,
    /// The footer's schema is an invalid flatbuffer
    InvalidFlatbufferSchema(polars_arrow_format::ipc::planus::Error),
    /// The file's schema does not contain fields
    MissingFields,
    /// The footer's dictionaries is an invalid flatbuffer
    InvalidFlatbufferDictionaries(polars_arrow_format::ipc::planus::Error),
    /// The block is an invalid flatbuffer
    InvalidFlatbufferBlock(polars_arrow_format::ipc::planus::Error),
    /// The dictionary message is an invalid flatbuffer
    InvalidFlatbufferMessage(polars_arrow_format::ipc::planus::Error),
    /// The message does not contain a header
    MissingMessageHeader,
    /// The message's header is an invalid flatbuffer
    InvalidFlatbufferHeader(polars_arrow_format::ipc::planus::Error),
    /// Relative positions in the file is < 0
    UnexpectedNegativeInteger,
    /// dictionaries can only contain dictionary messages; record batches can only contain records
    UnexpectedMessageType,
    /// RecordBatch messages do not contain buffers
    MissingMessageBuffers,
    /// The message's buffers is an invalid flatbuffer
    InvalidFlatbufferBuffers(polars_arrow_format::ipc::planus::Error),
    /// RecordBatch messages does not contain nodes
    MissingMessageNodes,
    /// The message's nodes is an invalid flatbuffer
    InvalidFlatbufferNodes(polars_arrow_format::ipc::planus::Error),
    /// The message's body length is an invalid flatbuffer
    InvalidFlatbufferBodyLength(polars_arrow_format::ipc::planus::Error),
    /// The message does not contain data
    MissingData,
    /// The message's data is an invalid flatbuffer
    InvalidFlatbufferData(polars_arrow_format::ipc::planus::Error),
    /// The version is an invalid flatbuffer
    InvalidFlatbufferVersion(polars_arrow_format::ipc::planus::Error),
    /// The compression is an invalid flatbuffer
    InvalidFlatbufferCompression(polars_arrow_format::ipc::planus::Error),
    /// The record contains a number of buffers that does not match the required number by the data type
    ExpectedBuffer,
    /// A buffer's size is smaller than the required for the number of elements
    InvalidBuffer {
        /// Declared number of elements in the buffer
        length: usize,
        /// The name of the `NativeType`
        type_name: &'static str,
        /// Bytes required for the `length` and `type`
        required_number_of_bytes: usize,
        /// The size of the IPC buffer
        buffer_length: usize,
    },
    /// A buffer's size is larger than the file size
    InvalidBuffersLength {
        /// number of bytes of all buffers in the record
        buffers_size: u64,
        /// the size of the file
        file_size: u64,
    },
    /// A bitmap's size is smaller than the required for the number of elements
    InvalidBitmap {
        /// Declared length of the bitmap
        length: usize,
        /// Number of bits on the IPC buffer
        number_of_bits: usize,
    },
    /// The dictionary is_delta is an invalid flatbuffer
    InvalidFlatbufferIsDelta(polars_arrow_format::ipc::planus::Error),
    /// The dictionary id is an invalid flatbuffer
    InvalidFlatbufferId(polars_arrow_format::ipc::planus::Error),
    /// Invalid dictionary id
    InvalidId {
        /// The requested dictionary id
        requested_id: i64,
    },
    /// Field id is not a dictionary
    InvalidIdDataType {
        /// The requested dictionary id
        requested_id: i64,
    },
    /// FixedSizeBinaryArray has invalid datatype.
    InvalidDataType,
}

impl Display for OutOfSpecKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
