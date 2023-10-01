use crate::error::Error;

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
    InvalidFlatbufferFooter(arrow_format::ipc::planus::Error),
    /// The file's footer does not contain record batches
    MissingRecordBatches,
    /// The footer's record batches is an invalid flatbuffer
    InvalidFlatbufferRecordBatches(arrow_format::ipc::planus::Error),
    /// The file's footer does not contain a schema
    MissingSchema,
    /// The footer's schema is an invalid flatbuffer
    InvalidFlatbufferSchema(arrow_format::ipc::planus::Error),
    /// The file's schema does not contain fields
    MissingFields,
    /// The footer's dictionaries is an invalid flatbuffer
    InvalidFlatbufferDictionaries(arrow_format::ipc::planus::Error),
    /// The block is an invalid flatbuffer
    InvalidFlatbufferBlock(arrow_format::ipc::planus::Error),
    /// The dictionary message is an invalid flatbuffer
    InvalidFlatbufferMessage(arrow_format::ipc::planus::Error),
    /// The message does not contain a header
    MissingMessageHeader,
    /// The message's header is an invalid flatbuffer
    InvalidFlatbufferHeader(arrow_format::ipc::planus::Error),
    /// Relative positions in the file is < 0
    UnexpectedNegativeInteger,
    /// dictionaries can only contain dictionary messages; record batches can only contain records
    UnexpectedMessageType,
    /// RecordBatch messages do not contain buffers
    MissingMessageBuffers,
    /// The message's buffers is an invalid flatbuffer
    InvalidFlatbufferBuffers(arrow_format::ipc::planus::Error),
    /// RecordBatch messages does not contain nodes
    MissingMessageNodes,
    /// The message's nodes is an invalid flatbuffer
    InvalidFlatbufferNodes(arrow_format::ipc::planus::Error),
    /// The message's body length is an invalid flatbuffer
    InvalidFlatbufferBodyLength(arrow_format::ipc::planus::Error),
    /// The message does not contain data
    MissingData,
    /// The message's data is an invalid flatbuffer
    InvalidFlatbufferData(arrow_format::ipc::planus::Error),
    /// The version is an invalid flatbuffer
    InvalidFlatbufferVersion(arrow_format::ipc::planus::Error),
    /// The compression is an invalid flatbuffer
    InvalidFlatbufferCompression(arrow_format::ipc::planus::Error),
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
    InvalidFlatbufferIsDelta(arrow_format::ipc::planus::Error),
    /// The dictionary id is an invalid flatbuffer
    InvalidFlatbufferId(arrow_format::ipc::planus::Error),
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

impl From<OutOfSpecKind> for Error {
    fn from(kind: OutOfSpecKind) -> Self {
        Error::OutOfSpec(format!("{kind:?}"))
    }
}

impl From<arrow_format::ipc::planus::Error> for Error {
    fn from(error: arrow_format::ipc::planus::Error) -> Self {
        Error::OutOfSpec(error.to_string())
    }
}
