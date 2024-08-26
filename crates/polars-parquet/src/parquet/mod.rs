#[macro_use]
pub mod error;
#[cfg(feature = "bloom_filter")]
pub mod bloom_filter;
pub mod compression;
pub mod encoding;
pub mod metadata;
pub mod page;
mod parquet_bridge;
pub mod read;
pub mod schema;
pub mod statistics;
pub mod types;
pub mod write;

use std::ops::Deref;

use parquet_format_safe as thrift_format;
use polars_utils::mmap::MemSlice;
pub use streaming_decompression::{fallible_streaming_iterator, FallibleStreamingIterator};

pub const HEADER_SIZE: u64 = PARQUET_MAGIC.len() as u64;
pub const FOOTER_SIZE: u64 = 8;
pub const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

/// The number of bytes read at the end of the parquet file on first read
const DEFAULT_FOOTER_READ_SIZE: u64 = 64 * 1024;

/// A copy-on-write buffer over bytes
#[derive(Debug, Clone)]
pub enum CowBuffer {
    Borrowed(MemSlice),
    Owned(Vec<u8>),
}

impl Deref for CowBuffer {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            CowBuffer::Borrowed(v) => v.deref(),
            CowBuffer::Owned(v) => v.deref(),
        }
    }
}

impl CowBuffer {
    pub fn to_mut(&mut self) -> &mut Vec<u8> {
        match self {
            CowBuffer::Borrowed(v) => {
                *self = Self::Owned(v.clone().to_vec());
                self.to_mut()
            },
            CowBuffer::Owned(v) => v,
        }
    }

    pub fn into_vec(self) -> Vec<u8> {
        match self {
            CowBuffer::Borrowed(v) => v.to_vec(),
            CowBuffer::Owned(v) => v,
        }
    }
}
