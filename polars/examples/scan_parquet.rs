use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let df = polars::scan_parquet!("examples/datasets/foods1.parquet")?
        .filter(col("category").eq(lit("vegetables")))
        .collect()?;
    println!("{df}");

    Ok(())
}
