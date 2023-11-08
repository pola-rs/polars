use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let df = LazyFrame::scan_parquet("../datasets/foods1.parquet", ScanArgsParquet::default())?
        .select([
            // select all columns
            all(),
            // and do some aggregations
            cols(["fats_g", "sugars_g"]).sum().name().suffix("_summed"),
        ])
        .collect()?;

    println!("{}", df);
    Ok(())
}
