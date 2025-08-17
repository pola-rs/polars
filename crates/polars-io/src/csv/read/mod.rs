//! Functionality for reading CSV files.
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
//!     CsvReadOptions::default()
//!         .with_has_header(true)
//!         .try_into_reader_with_file_path(Some("example.csv".into()))?
//!         .finish()
//! }
//! ```

pub mod buffer;
mod options;
mod parser;
mod read_impl;
mod reader;
pub mod schema_inference;
mod splitfields;
mod utils;

pub use options::{CommentPrefix, CsvEncoding, CsvParseOptions, CsvReadOptions, NullValues};
pub use parser::{count_rows, count_rows_from_slice, count_rows_from_slice_par};
pub use read_impl::batched::{BatchedCsvReader, OwnedBatchedCsvReader};
pub use reader::CsvReader;
pub use schema_inference::infer_file_schema;

pub mod _csv_read_internal {
    pub use super::buffer::validate_utf8;
    pub use super::options::NullValuesCompiled;
    pub use super::parser::CountLines;
    pub use super::read_impl::{cast_columns, find_starting_point, read_chunk};
    pub use super::reader::prepare_csv_schema;
}
