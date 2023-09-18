//! Convert data between the Arrow memory format and JSON line-delimited records.

#[cfg(feature = "io_json_read")]
pub mod read;
#[cfg(feature = "io_json_write")]
pub mod write;
