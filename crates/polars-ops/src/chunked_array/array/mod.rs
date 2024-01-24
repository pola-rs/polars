#[cfg(feature = "array_any_all")]
mod any_all;
mod count;
mod get;
mod join;
mod min_max;
mod namespace;
mod sum_mean;

pub use namespace::ArrayNameSpace;
use polars_core::prelude::*;

pub trait AsArray {
    fn as_array(&self) -> &ArrayChunked;
}

impl AsArray for ArrayChunked {
    fn as_array(&self) -> &ArrayChunked {
        self
    }
}
