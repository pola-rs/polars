use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let df = polars::read_csv!("examples/datasets/foods1.csv")?;

    println!("{df}");

    Ok(())
}
