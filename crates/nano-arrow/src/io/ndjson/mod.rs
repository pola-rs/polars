//! APIs to read from and write to NDJSON

#[cfg(feature = "io_json_read")]
pub mod read;
#[cfg(feature = "io_json_write")]
pub mod write;
