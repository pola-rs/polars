pub mod anonymous_scan;
#[cfg(feature = "csv")]
pub mod csv;
pub(crate) mod file_list_reader;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "json")]
pub mod ndjson;
#[cfg(feature = "parquet")]
pub mod parquet;
use file_list_reader::*;
