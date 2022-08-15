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
//! fn example(df: &mut DataFrame) -> Result<()> {
//!     let mut file = File::create("example.csv").expect("could not create file");
//!
//!     CsvWriter::new(&mut file)
//!     .has_header(true)
//!     .with_delimiter(b',')
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
//! fn example() -> Result<DataFrame> {
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
#[cfg(not(feature = "private"))]
pub(crate) mod utils;
#[cfg(feature = "private")]
pub mod utils;
mod write;
pub(super) mod write_impl;

use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use polars_core::prelude::*;
#[cfg(feature = "temporal")]
use polars_time::prelude::*;
#[cfg(feature = "temporal")]
use rayon::prelude::*;
pub use read::{CsvEncoding, CsvReader, NullValues};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use write::CsvWriter;

use crate::aggregations::ScanAggregation;
use crate::csv::read_impl::{cast_columns, CoreReader};
use crate::csv::utils::get_reader_bytes;
use crate::mmap::MmapBytesReader;
use crate::predicates::PhysicalIoExpr;
use crate::utils::resolve_homedir;
use crate::{RowCount, SerReader, SerWriter};

#[cfg(test)]
mod tests {
    extern crate test;

    use polars_core::prelude::*;
    use rand::prelude::*;
    use test::Bencher;

    use crate::csv::CsvWriter;
    use crate::SerWriter;

    #[bench]
    fn benchmark_write_csv_f32(b: &mut Bencher) -> Result<()> {
        // NOTE: This benchmark can be run by executing ``$ cargo bench -p polars-io``
        // from within /polars/polars/
        const N: usize = 10_000_000;

        let vec: Vec<f32> = (0..N).map(|_| thread_rng().next_u32() as f32).collect();

        let mut df = df![
            "random" => vec.as_slice(),
        ]?;

        let mut buffer: Vec<u8> = Vec::new();

        b.iter(|| CsvWriter::new(&mut buffer).finish(&mut df));

        Ok(())
    }
}
