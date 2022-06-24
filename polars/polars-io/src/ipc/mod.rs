use super::*;

#[cfg(feature = "ipc")]
mod ipc_file;

#[cfg(feature = "ipc_streaming")]
mod ipc_stream;

#[cfg(feature = "ipc")]
pub use crate::ipc::ipc_file::{IpcCompression, IpcReader, IpcWriter, IpcWriterOption};

#[cfg(feature = "ipc_streaming")]
pub use ipc_stream::*;
