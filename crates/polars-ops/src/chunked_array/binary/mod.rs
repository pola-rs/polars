mod namespace;

pub use namespace::*;
use polars_core::prelude::*;

pub trait AsBinary {
    fn as_binary(&self) -> &BinaryChunked;
}

impl AsBinary for BinaryChunked {
    fn as_binary(&self) -> &BinaryChunked {
        self
    }
}
