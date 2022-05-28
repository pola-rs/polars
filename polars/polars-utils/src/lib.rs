#![cfg_attr(docsrs, feature(doc_cfg))]
pub mod arena;
pub mod contention_pool;
mod error;
mod functions;
pub mod mem;
pub mod sort;

pub use functions::*;

#[cfg(not(feature = "bigidx"))]
pub type IdxSize = u32;
#[cfg(feature = "bigidx")]
pub type IdxSize = u64;
