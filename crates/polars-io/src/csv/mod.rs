//! # (De)serializing CSV files
//!
//! ## Maximal performance
//! Currently [CsvReader::new](CsvReader::new) has an extra copy. If you want optimal performance in CSV parsing/
//! reading, it is advised to use [CsvReader::from_path](CsvReader::from_path).
//!
//! ## Write a DataFrame to a csv file.
//!
//! ## Example
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
//!     .include_header(true)
//!     .with_separator(b',')
//!     .finish(df)
//! }
//! ```
//!
//! ## Read a csv file to a DataFrame
//!
//! ## Example
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> PolarsResult<DataFrame> {
//!     // always prefer `from_path` as that is fastest.
//!     CsvReader::from_path("iris_csv")?
//!             .has_header(true)
//!             .finish()
//! }
//! ```
//!
pub(crate) mod buffer;
pub(crate) mod parser;
pub mod read_impl;

mod read;
pub(super) mod splitfields;
pub mod utils;
mod write;
pub(super) mod write_impl;

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use polars_core::prelude::*;
#[cfg(feature = "temporal")]
use polars_time::prelude::*;
#[cfg(feature = "temporal")]
use rayon::prelude::*;
pub use read::{CommentPrefix, CsvEncoding, CsvReader, NullValues};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use write::{BatchedWriter, CsvWriter, QuoteStyle};
pub use write_impl::SerializeOptions;

use crate::csv::read_impl::CoreReader;
use crate::mmap::MmapBytesReader;
use crate::predicates::PhysicalIoExpr;
use crate::utils::{get_reader_bytes, resolve_homedir};
use crate::{RowCount, SerReader, SerWriter};
