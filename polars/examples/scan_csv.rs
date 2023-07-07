use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let df = polars::scan_csv!("examples/datasets/foods1.csv", has_header = true)?.collect()?;
    println!("{df}");
    Ok(())
}
