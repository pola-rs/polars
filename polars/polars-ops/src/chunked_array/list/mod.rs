use polars_core::prelude::*;

mod count;
#[cfg(feature = "hash")]
pub(crate) mod hash;
mod min_max;
mod namespace;
mod sum_mean;
#[cfg(feature = "list_to_struct")]
mod to_struct;

#[cfg(feature = "list_count")]
pub use count::*;
#[cfg(not(feature = "list_count"))]
use count::*;
pub use namespace::*;
#[cfg(feature = "list_to_struct")]
pub use to_struct::*;

pub trait AsList {
    fn as_list(&self) -> &ListChunked;
}

impl AsList for ListChunked {
    fn as_list(&self) -> &ListChunked {
        self
    }
}
