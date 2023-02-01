#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "nightly", feature(build_hasher_simple_hash_one))]

pub mod arena;
pub mod atomic;
pub mod cell;
pub mod contention_pool;
mod error;
mod functions;
mod hash;
pub mod mem;
pub mod slice;
pub mod sort;
pub mod sync;
#[cfg(feature = "sysinfo")]
pub mod sys;
pub mod unwrap;

pub use functions::*;
pub use hash::HashSingle;

#[cfg(not(feature = "bigidx"))]
pub type IdxSize = u32;
#[cfg(feature = "bigidx")]
pub type IdxSize = u64;

pub mod macros;
#[cfg(target_family = "wasm")]
pub mod wasm;
