use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let df = LazyFrame::scan_parquet("../datasets/foods1.parquet", ParquetOptions::default())?
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
