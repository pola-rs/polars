#![cfg_attr(
    all(target_arch = "aarch64", feature = "nightly"),
    feature(stdarch_aarch64_prefetch)
)]
#![cfg_attr(feature = "nightly", feature(core_intrinsics))] // For algebraic ops.
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
pub mod abs_diff;
pub mod algebraic_ops;
pub mod arena;
pub mod binary_search;
pub mod cache;
pub mod cardinality_sketch;
pub mod cell;
pub mod chunks;
pub mod clmul;
mod config;
pub mod cpuid;
pub mod error;
pub mod floor_divmod;
pub mod functions;
pub mod hashing;
pub mod idx_map;
pub mod idx_vec;
pub mod mem;
pub mod min_max;
pub mod pl_str;
pub mod priority;
pub mod slice;
pub mod sort;
pub mod sync;
#[cfg(feature = "sysinfo")]
pub mod sys;
pub mod total_ord;

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

#[cfg(feature = "python")]
pub mod python_function;

#[cfg(feature = "serde")]
pub mod pl_serialize;
