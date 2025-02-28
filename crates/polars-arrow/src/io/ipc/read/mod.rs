//! APIs to read Arrow's IPC format.
//!
//! The two important structs here are the [`FileReader`](reader::FileReader),
//! which provides arbitrary access to any of its messages, and the
//! [`StreamReader`](stream::StreamReader), which only supports reading
//! data in the order it was written in.
use crate::array::Array;

mod array;
mod common;
mod deserialize;
mod error;
pub(crate) mod file;
#[cfg(feature = "io_flight")]
mod flight;
mod read_basic;
mod reader;
mod schema;
mod stream;

pub(crate) use common::first_dict_field;
pub use common::{prepare_projection, ProjectionInfo};
pub use error::OutOfSpecKind;
pub use file::{
    deserialize_footer, get_row_count, get_row_count_from_blocks, read_batch,
    read_file_dictionaries, read_file_metadata, FileMetadata,
};
use polars_utils::aliases::PlHashMap;
pub use reader::FileReader;
pub use schema::deserialize_schema;
pub use stream::{read_stream_metadata, StreamMetadata, StreamReader, StreamState};

/// how dictionaries are tracked in this crate
pub type Dictionaries = PlHashMap<i64, Box<dyn Array>>;

pub(crate) type Node<'a> = arrow_format::ipc::FieldNodeRef<'a>;
pub(crate) type IpcBuffer<'a> = arrow_format::ipc::BufferRef<'a>;
pub(crate) type Compression<'a> = arrow_format::ipc::BodyCompressionRef<'a>;
pub(crate) type Version = arrow_format::ipc::MetadataVersion;

#[cfg(feature = "io_flight")]
pub use flight::*;

pub trait SendableIterator: Send + Iterator {}

impl<T: Iterator + Send> SendableIterator for T {}
