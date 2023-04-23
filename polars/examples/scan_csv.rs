use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let df = polars::scan_csv!("examples/datasets/foods1.csv")?.collect()?;
    println!("{df}");

    Ok(())
}
