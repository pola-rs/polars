use polars_core::prelude::*;

#[cfg(feature = "hash")]
pub(crate) mod hash;
#[cfg(feature = "list")]
#[cfg_attr(docsrs, doc(cfg(feature = "list")))]
mod namespace;
#[cfg(feature = "list_to_struct")]
mod to_struct;

#[cfg(feature = "list")]
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
