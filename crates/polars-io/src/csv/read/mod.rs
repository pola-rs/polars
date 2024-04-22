//! Functionality for reading CSV files.
//!
//! Note: currently, [CsvReader::new](CsvReader::new) has an extra copy. If you want optimal
//! performance in CSV parsing/reading, it is advised to use
//! [CsvReader::from_path](CsvReader::from_path).
//!
//! # Examples
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> PolarsResult<DataFrame> {
//!     // Always prefer `from_path` as it is fastest.
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

pub use options::{CommentPrefix, CsvEncoding, CsvParserOptions, NullValues};
pub use parser::count_rows;
pub use read_impl::batched_mmap::{BatchedCsvReaderMmap, OwnedBatchedCsvReaderMmap};
pub use read_impl::batched_read::{BatchedCsvReaderRead, OwnedBatchedCsvReader};
pub use reader::CsvReader;
pub use utils::{infer_file_schema, is_compressed};
