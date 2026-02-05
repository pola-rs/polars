#![cfg_attr(
    all(target_arch = "aarch64", feature = "nightly"),
    feature(stdarch_aarch64_prefetch)
)]
#![cfg_attr(feature = "nightly", feature(core_intrinsics))] // For algebraic ops, select_unpredictable.
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(docsrs, feature(doc_cfg))]
pub mod abs_diff;
pub mod algebraic_ops;
pub mod arena;
pub mod array;
pub mod binary_search;
pub mod cache;
pub mod cardinality_sketch;
pub mod cell;
pub mod chunks;
pub mod clmul;
mod config;
pub mod small_bytes;
pub use config::check_allow_importing_interval_as_struct;
pub mod arg_min_max;
pub mod bool;
pub mod cpuid;
pub mod error;
pub mod float16;
pub mod floor_divmod;
pub mod functions;
pub mod hashing;
pub mod ideal_morsel_size;
pub mod idx_map;
pub mod idx_vec;
pub mod live_timer;
pub mod marked_usize;
pub mod mem;
pub mod min_max;
pub mod order_statistic_tree;
pub mod parma;
pub mod pl_path;
mod pl_ref_str;
pub mod pl_str;
pub mod priority;
pub mod regex_cache;
pub mod relaxed_cell;
pub mod reuse_vec;
pub mod row_counter;
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
pub mod with_drop;
pub use functions::*;
pub mod compression;
pub mod file;

pub mod aliases;
pub mod fixedringbuffer;
pub mod fmt;
pub mod itertools;
pub mod macros;
pub mod option;
pub mod vec;

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
pub use either;
pub use idx_vec::UnitVec;
pub mod chunked_bytes_cursor;
pub mod concat_vec;
