use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let df = polars::read_csv!("examples/datasets/foods1.csv")?
        .lazy()
        .filter(col("category").eq(lit("vegetables")))
        .collect()?;
    println!("{df}");

    Ok(())
}
