pub(super) mod anonymous_scan;
#[cfg(feature = "csv")]
pub(super) mod csv;
pub(super) mod file_list_reader;
#[cfg(feature = "ipc")]
pub(super) mod ipc;
#[cfg(feature = "json")]
pub(super) mod ndjson;
#[cfg(feature = "parquet")]
pub(super) mod parquet;

use file_list_reader::*;
