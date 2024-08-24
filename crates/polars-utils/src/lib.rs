#![cfg_attr(
    all(target_arch = "aarch64", feature = "nightly"),
    feature(stdarch_aarch64_prefetch)
)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
pub mod abs_diff;
pub mod arena;
pub mod atomic;
pub mod binary_search;
pub mod cache;
pub mod cell;
pub mod clmul;
pub mod contention_pool;
pub mod cpuid;
mod error;
pub mod floor_divmod;
pub mod foreign_vec;
pub mod functions;
pub mod hashing;
pub mod idx_vec;
pub mod mem;
pub mod min_max;
pub mod priority;
pub mod slice;
pub mod sort;
pub mod sync;
#[cfg(feature = "sysinfo")]
pub mod sys;
pub mod total_ord;
pub mod unwrap;

pub use functions::*;

pub mod aliases;
pub mod fixedringbuffer;
pub mod fmt;
pub mod itertools;
pub mod macros;
pub mod vec;
#[cfg(target_family = "wasm")]
pub mod wasm;

pub mod float;
pub mod index;
pub mod io;
#[cfg(feature = "mmap")]
pub mod mmap;
pub mod nulls;
pub mod ord;
pub mod partitioned;

pub use index::{IdxSize, NullableIdxSize};
pub use io::*;
