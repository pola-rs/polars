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
pub use write_impl::{CsvSerializer, UTF8_BOM, csv_header};
pub use writer::{BatchedWriter, CsvWriter};

#[cfg(test)]
mod test {
    use polars_core::df;
    use polars_core::prelude::*;

    use super::*;
    use crate::SerWriter;

    #[test]
    fn scientific_with_decimal_comma_writes_each_value_once() {
        let mut df = df!["a" => [1.5f64, 2.5, 3.5]].unwrap();
        let mut buf = Vec::new();
        CsvWriter::new(&mut buf)
            .with_float_scientific(Some(true))
            .with_decimal_comma(true)
            .with_separator(b';')
            .finish(&mut df)
            .unwrap();
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            "a\n1,5e0\n2,5e0\n3,5e0\n"
        );
    }
}
