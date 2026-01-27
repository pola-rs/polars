pub mod multi_scan;

pub mod batch;
#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "ipc")]
pub mod ipc;
#[cfg(feature = "scan_lines")]
pub mod lines;
#[cfg(any(feature = "json", feature = "scan_lines"))]
pub mod ndjson;
#[cfg(feature = "parquet")]
pub mod parquet;
