#![cfg_attr(docsrs, feature(doc_auto_cfg))]
pub mod arena;
pub mod atomic;
pub mod cell;
pub mod contention_pool;
mod error;
mod functions;
pub mod mem;
pub mod slice;
pub mod sort;
pub mod sync;
#[cfg(feature = "sysinfo")]
pub mod sys;
pub mod unwrap;

pub use functions::*;

#[cfg(not(feature = "bigidx"))]
pub type IdxSize = u32;
#[cfg(feature = "bigidx")]
pub type IdxSize = u64;

pub mod aliases;
pub mod fmt;
pub mod iter;
pub mod macros;
pub mod vec;
#[cfg(target_family = "wasm")]
pub mod wasm;
