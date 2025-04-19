pub mod multi_file_reader;

pub mod batch;
#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "json")]
pub mod ndjson;
#[cfg(feature = "parquet")]
pub mod parquet;
