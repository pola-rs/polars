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

pub mod builder;
mod options;
mod parser;
mod read_impl;
mod reader;
pub mod schema_inference;
mod splitfields;
pub mod streaming;
mod utils;

pub use options::{CommentPrefix, CsvEncoding, CsvParseOptions, CsvReadOptions, NullValues};
pub use parser::{SplitLines, count_rows, count_rows_from_slice_par};
pub use reader::CsvReader;
pub use streaming::read_until_start_and_infer_schema;

pub mod _csv_read_internal {
    pub use super::builder::validate_utf8;
    pub use super::options::{CommentPrefix, NullValuesCompiled};
    pub use super::parser::{CountLines, SplitLines, is_comment_line};
    pub use super::read_impl::{cast_columns, read_chunk};
    pub use super::reader::prepare_csv_schema;
}
