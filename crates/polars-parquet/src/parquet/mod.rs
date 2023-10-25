#[macro_use]
pub mod error;
#[cfg(feature = "bloom_filter")]
pub mod bloom_filter;
pub mod compression;
pub mod deserialize;
pub mod encoding;
pub mod indexes;
pub mod metadata;
pub mod page;
mod parquet_bridge;
pub mod read;
pub mod schema;
pub mod statistics;
pub mod types;
pub mod write;

use parquet_format_safe as thrift_format;
pub use streaming_decompression::{fallible_streaming_iterator, FallibleStreamingIterator};

const HEADER_SIZE: u64 = PARQUET_MAGIC.len() as u64;
const FOOTER_SIZE: u64 = 8;
const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

/// The number of bytes read at the end of the parquet file on first read
const DEFAULT_FOOTER_READ_SIZE: u64 = 64 * 1024;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    pub fn get_path() -> PathBuf {
        let dir = env!("CARGO_MANIFEST_DIR");

        PathBuf::from(dir).join("testing/parquet-testing/data")
    }
}
