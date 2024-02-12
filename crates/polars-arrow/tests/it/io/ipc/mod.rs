mod common;
mod read;
mod write;

pub use common::read_gzip_json;

#[cfg(feature = "io_ipc_write_async")]
mod write_stream_async;

#[cfg(feature = "io_ipc_write_async")]
mod write_file_async;

#[cfg(feature = "io_ipc_read_async")]
mod read_stream_async;

#[cfg(feature = "io_ipc_read_async")]
mod read_file_async;

mod mmap;
