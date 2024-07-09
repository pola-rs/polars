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

pub use options::{ParallelStrategy, ParquetOptions};
#[cfg(feature = "cloud")]
pub use reader::ParquetAsyncReader;
pub use reader::{BatchedParquetReader, ParquetReader};
pub use utils::materialize_empty_df;
