use polars_core::prelude::*;

mod count;
mod dispersion;
mod get;
mod min_max;
mod namespace;
#[cfg(feature = "list_sets")]
mod sets;
mod sum_mean;
#[cfg(feature = "list_to_struct")]
mod to_struct;

#[cfg(feature = "list_count")]
pub use count::*;
#[cfg(not(feature = "list_count"))]
use count::*;
pub use get::*;
pub use namespace::*;
#[cfg(feature = "list_sets")]
pub use sets::*;
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
