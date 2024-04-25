//! Functionality for reading CSV files.
//!
//! Note: currently, `CsvReader::new` has an extra copy. If you want optimal performance,
//! it is advised to use [`CsvReader::from_path`] instead.
//!
//! # Examples
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> PolarsResult<DataFrame> {
//!     // Prefer `from_path` over `new` as it is faster.
//!     CsvReader::from_path("example.csv")?
//!         .has_header(true)
//!         .finish()
//! }
//! ```

mod buffer;
mod options;
mod parser;
mod read_impl;
mod reader;
mod splitfields;
mod utils;

pub use options::{CommentPrefix, CsvEncoding, CsvReaderOptions, NullValues};
pub use parser::count_rows;
pub use read_impl::batched_mmap::{BatchedCsvReaderMmap, OwnedBatchedCsvReaderMmap};
pub use read_impl::batched_read::{BatchedCsvReaderRead, OwnedBatchedCsvReader};
pub use reader::CsvReader;
pub use utils::{infer_file_schema, is_compressed};
