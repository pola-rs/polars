use super::*;

#[cfg(feature = "ipc")]
mod ipc_file;

#[cfg(feature = "ipc_streaming")]
mod ipc_stream;
mod mmap;

#[cfg(feature = "ipc")]
pub use ipc_file::{BatchedWriter, IpcCompression, IpcReader, IpcWriter, IpcWriterOption};
#[cfg(feature = "ipc_streaming")]
pub use ipc_stream::*;
