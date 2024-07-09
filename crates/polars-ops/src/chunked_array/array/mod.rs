#[cfg(feature = "array_any_all")]
mod any_all;
mod count;
mod dispersion;
mod get;
mod join;
mod min_max;
mod namespace;
mod sum_mean;
#[cfg(feature = "array_to_struct")]
mod to_struct;

pub use namespace::ArrayNameSpace;
use polars_core::prelude::*;
#[cfg(feature = "array_to_struct")]
pub use to_struct::*;

pub trait AsArray {
    fn as_array(&self) -> &ArrayChunked;
}

impl AsArray for ArrayChunked {
    fn as_array(&self) -> &ArrayChunked {
        self
    }
}
