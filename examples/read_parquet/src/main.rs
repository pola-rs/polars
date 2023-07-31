use polars::prelude::*;

fn main() -> PolarsResult<()> {
    csv_to_parquet();

    let df = LazyFrame::scan_parquet("../datasets/foods1.parquet", ScanArgsParquet::default())?
        .select([
            // select all columns
            all(),
            // and do some aggregations
            cols(["fats_g", "sugars_g"]).sum().suffix("_summed"),
        ])
        .collect()?;

    dbg!(df);
    Ok(())
}

fn csv_to_df() -> DataFrame {
    CsvReader::from_path("../datasets/foods1.csv")
        .unwrap()
        .finish()
        .unwrap()
}

fn csv_to_parquet() {
    let mut df = csv_to_df();
    let mut file = std::fs::File::create("../datasets/foods1.parquet").unwrap();
    ParquetWriter::new(&mut file).finish(&mut df).unwrap();
}
