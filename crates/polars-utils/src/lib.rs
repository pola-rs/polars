#![cfg_attr(
    all(target_arch = "aarch64", feature = "nightly"),
    feature(stdarch_aarch64_prefetch)
)]
#![allow(stable_features)] // float_algebraic is stable in nightly but not on stable yet
#![cfg_attr(feature = "nightly", feature(float_algebraic))]
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod abs_diff;
pub mod algebraic_ops;
pub mod aliases;
pub mod arc;
pub mod arena;
pub mod arg_min_max;
pub mod array;
pub mod binary_search;
pub mod bool;
pub mod broadcast;
pub mod cache;
pub mod calc_morsel_split;
pub mod cardinality_sketch;
pub mod cell;
pub mod chunked_bytes_cursor;
pub mod chunks;
pub mod clmul;
pub mod collection;
pub mod compression;
pub mod concat_vec;
pub mod cpuid;
pub mod error;
pub mod file;
pub mod fixedringbuffer;
pub mod float;
pub mod float16;
pub mod floor_divmod;
pub mod fmt;
pub mod hashing;
pub mod idx_map;
pub mod idx_vec;
pub mod index;
pub mod io;
pub mod itertools;
pub mod kahan_sum;
pub mod levenshtein;
pub mod live_timer;
pub mod macros;
pub mod marked_usize;
pub mod mem;
pub mod min_max;
pub mod nulls;
pub mod option;
pub mod order_statistic_tree;
pub mod parma;
pub mod pl_path;
mod pl_ref_str;
pub mod pl_str;
pub mod priority;
pub mod range;
pub mod regex_cache;
pub mod relaxed_cell;
pub mod row_counter;
pub mod scratch_vec;
pub mod select;
pub mod slice;
pub mod slice_enum;
pub mod small_bytes;
pub mod sort;
pub mod sparse_init_vec;
pub mod sync;
pub mod tick_counter;
pub mod total_ord;
pub mod unique_id;
pub mod vec;
pub mod with_drop;

#[cfg(feature = "async-utils")]
pub mod async_utils;
#[cfg(feature = "mmap")]
pub mod mmap;
#[cfg(feature = "serde")]
pub mod pl_serialize;
#[cfg(feature = "python")]
pub mod python_convert_registry;
#[cfg(feature = "python")]
pub mod python_function;
#[cfg(feature = "python")]
pub mod python_interns;
#[cfg(feature = "python")]
pub mod python_thread_pool;
#[cfg(feature = "sysinfo")]
pub mod sys;

pub use idx_vec::UnitVec;
pub use index::{IdxSize, NullableIdxSize};
pub use pl_str::unique_column_name;
