#[cfg(feature = "extract_jsonpath")]
mod json_path;
#[cfg(feature = "strings")]
mod namespace;

#[cfg(feature = "extract_jsonpath")]
pub use json_path::*;
#[cfg(feature = "strings")]
pub use namespace::*;
use polars_core::prelude::*;

pub trait AsUtf8 {
    fn as_utf8(&self) -> &Utf8Chunked;
}

impl AsUtf8 for Utf8Chunked {
    fn as_utf8(&self) -> &Utf8Chunked {
        self
    }
}
