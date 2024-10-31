//! Functionality for reading Apache Parquet files.
//!
//! # Examples
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> PolarsResult<DataFrame> {
//!     let r = File::open("example.parquet").unwrap();
//!     let reader = ParquetReader::new(r);
//!     reader.finish()
//! }
//! ```

#[cfg(feature = "cloud")]
mod async_impl;
mod mmap;
mod options;
mod predicates;
mod read_impl;
mod reader;
mod to_metadata;
mod utils;

const ROW_COUNT_OVERFLOW_ERR: PolarsError = PolarsError::ComputeError(ErrString::new_static(
    "\
Parquet file produces more than pow(2, 32) rows; \
consider compiling with polars-bigidx feature (polars-u64-idx package on python), \
or set 'streaming'",
));

pub use options::{ParallelStrategy, ParquetOptions};
use polars_error::{ErrString, PolarsError};
pub use read_impl::{create_sorting_map, try_set_sorted_flag};
#[cfg(feature = "cloud")]
pub use reader::ParquetAsyncReader;
pub use reader::{BatchedParquetReader, ParquetReader};
pub use utils::materialize_empty_df;

pub mod _internal {
    pub use super::mmap::to_deserializer;
    pub use super::predicates::read_this_row_group;
    pub use super::read_impl::{calc_prefilter_cost, PrefilterMaskSetting};
    pub use super::utils::ensure_matching_dtypes_if_found;
}
