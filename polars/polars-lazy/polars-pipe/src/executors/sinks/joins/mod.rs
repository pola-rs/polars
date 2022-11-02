#[cfg(feature = "cross_join")]
mod cross;
mod inner;

#[cfg(feature = "cross_join")]
pub(crate) use cross::*;
