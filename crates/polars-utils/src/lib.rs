#![cfg_attr(
    all(target_arch = "aarch64", feature = "nightly"),
    feature(stdarch_aarch64_prefetch)
)]
#![cfg_attr(feature = "nightly", feature(core_intrinsics))] // For algebraic ops.
#![cfg_attr(feature = "nightly", feature(select_unpredictable))] // For branchless programming.
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
pub mod idx_mapper;
pub mod idx_vec;
pub mod mem;
pub mod min_max;
pub mod pl_str;
pub mod priority;
pub mod regex_cache;
pub mod select;
pub mod slice;
pub mod slice_enum;
pub mod sort;
pub mod sparse_init_vec;
pub mod sync;
#[cfg(feature = "sysinfo")]
pub mod sys;
pub mod total_ord;
pub mod unique_id;

pub use functions::*;
pub mod file;

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
pub mod partitioned;

pub use index::{IdxSize, NullableIdxSize};
pub use io::*;
pub use pl_str::unique_column_name;

#[cfg(feature = "python")]
pub mod python_function;

#[cfg(feature = "python")]
pub mod python_convert_registry;

#[cfg(feature = "serde")]
pub mod pl_serialize;

pub mod kahan_sum;
