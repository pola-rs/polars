//! Functionality for writing CSV files.
//!
//! # Examples
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example(df: &mut DataFrame) -> PolarsResult<()> {
//!     let mut file = File::create("example.csv").expect("could not create file");
//!
//!     CsvWriter::new(&mut file)
//!         .include_header(true)
//!         .with_separator(b',')
//!         .finish(df)
//! }
//! ```

mod options;
mod write_impl;
mod writer;

pub use options::{CsvWriterOptions, QuoteStyle, SerializeOptions};
pub use writer::{BatchedWriter, CsvWriter};
