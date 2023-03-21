use super::*;

#[cfg(feature = "ipc")]
mod ipc_file;

#[cfg(feature = "ipc_streaming")]
mod ipc_stream;
mod mmap;
#[cfg(feature = "ipc")]
mod write;
#[cfg(all(feature = "async", feature = "ipc"))]
mod write_async;

#[cfg(feature = "ipc")]
pub use ipc_file::IpcReader;
#[cfg(feature = "ipc_streaming")]
pub use ipc_stream::*;
pub use write::{BatchedWriter, IpcCompression, IpcWriter, IpcWriterOption};
