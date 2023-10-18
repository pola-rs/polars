//! APIs to read from and write to Arrow's IPC format.
//!
//! Inter-process communication is a method through which different processes
//! share and pass data between them. Its use-cases include parallel
//! processing of chunks of data across different CPU cores, transferring
//! data between different Apache Arrow implementations in other languages and
//! more. Under the hood Apache Arrow uses [FlatBuffers](https://google.github.io/flatbuffers/)
//! as its binary protocol, so every Arrow-centered streaming or serialiation
//! problem that could be solved using FlatBuffers could probably be solved
//! using the more integrated approach that is exposed in this module.
//!
//! [Arrow's IPC protocol](https://arrow.apache.org/docs/format/Columnar.html#serialization-and-interprocess-communication-ipc)
//! allows only batch or dictionary columns to be passed
//! around due to its reliance on a pre-defined data scheme. This constraint
//! provides a large performance gain because serialized data will always have a
//! known structutre, i.e. the same fields and datatypes, with the only variance
//! being the number of rows and the actual data inside the Batch. This dramatically
//! increases the deserialization rate, as the bytes in the file or stream are already
//! structured "correctly".
//!
//! Reading and writing IPC messages is done using one of two variants - either
//! [`FileReader`](read::FileReader) <-> [`FileWriter`](struct@write::FileWriter) or
//! [`StreamReader`](read::StreamReader) <-> [`StreamWriter`](struct@write::StreamWriter).
//! These two variants wrap a type `T` that implements [`Read`](std::io::Read), and in
//! the case of the `File` variant it also implements [`Seek`](std::io::Seek). In
//! practice it means that `File`s can be arbitrarily accessed while `Stream`s are only
//! read in certain order - the one they were written in (first in, first out).
mod compression;
mod endianness;

pub mod append;
pub mod read;
pub mod write;

const ARROW_MAGIC_V1: [u8; 4] = [b'F', b'E', b'A', b'1'];
const ARROW_MAGIC_V2: [u8; 6] = [b'A', b'R', b'R', b'O', b'W', b'1'];
pub(crate) const CONTINUATION_MARKER: [u8; 4] = [0xff; 4];

/// Struct containing `dictionary_id` and nested `IpcField`, allowing users
/// to specify the dictionary ids of the IPC fields when writing to IPC.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IpcField {
    /// optional children
    pub fields: Vec<IpcField>,
    /// dictionary id
    pub dictionary_id: Option<i64>,
}

/// Struct containing fields and whether the file is written in little or big endian.
#[derive(Debug, Clone, PartialEq)]
pub struct IpcSchema {
    /// The fields in the schema
    pub fields: Vec<IpcField>,
    /// Endianness of the file
    pub is_little_endian: bool,
}
