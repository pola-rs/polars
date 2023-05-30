use polars::lazy::prelude::*;
use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let a = Series::new("a", [1, 2, 3, 4]);
    let b = Series::new("b", ["one", "two", "three", "four"]);
    let df = DataFrame::new(vec![a, b])?
        .lazy()
        .filter(col("b").str().starts_with(lit("t")))
        .collect()?;
    println!("{df:?}");
    Ok(())
}
