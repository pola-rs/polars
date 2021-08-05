//! tests that require parsing a csv
//!

use crate::csv::CsvReader;
use crate::SerReader;
use polars_core::prelude::*;

#[test]
fn test_filter() -> Result<()> {
    let path = "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
    let df = CsvReader::from_path(path)?.finish()?;

    let out = df.filter(&df.column("fats_g")?.gt(4))?;

    // this fails if all columns are not equal.
    dbg!(out);

    Ok(())
}
