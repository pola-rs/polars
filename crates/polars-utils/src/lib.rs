#![cfg_attr(docsrs, feature(doc_auto_cfg))]
pub mod abs_diff;
pub mod arena;
pub mod atomic;
pub mod cache;
pub mod cell;
pub mod contention_pool;
mod error;
pub mod functions;
pub mod hashing;
pub mod idx_vec;
pub mod mem;
pub mod min_max;
pub mod signed_divmod;
pub mod slice;
pub mod sort;
pub mod sync;
#[cfg(feature = "sysinfo")]
pub mod sys;
pub mod total_ord;
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

pub mod float;
pub mod index;
pub mod io;
pub mod nulls;
pub mod ord;

pub use io::open_file;
