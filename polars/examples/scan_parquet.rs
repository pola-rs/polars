use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let df = polars::scan_parquet!("examples/datasets/foods1.parquet")?
        .filter(col("category").eq(lit("vegetables")))
        .collect()?;
    println!("{df}");

    with_options()?;

    Ok(())
}

fn with_options() -> PolarsResult<()> {
    let df = polars::scan_parquet!(
        "examples/datasets/foods1.parquet",
        rechunk = true,
        low_memory = true,
        n_rows = 1,
        parallel = "auto"
    )?
    .filter(col("category").eq(lit("vegetables")))
    .collect()?;
    println!("{df}");

    Ok(())
}
