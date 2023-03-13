//! tests that require parsing a csv
//!

use polars_core::prelude::*;

use crate::csv::CsvReader;
use crate::SerReader;

#[test]
fn test_filter() -> PolarsResult<()> {
    let path = "../../examples/datasets/foods1.csv";
    let df = CsvReader::from_path(path)?.finish()?;

    let out = df.filter(&df.column("fats_g")?.gt(4)?)?;

    // this fails if all columns are not equal.
    println!("{out}");

    Ok(())
}
