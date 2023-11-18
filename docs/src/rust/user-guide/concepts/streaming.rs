use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:streaming]
    let q = LazyCsvReader::new("docs/data/iris.csv")
        .has_header(true)
        .finish()?
        .filter(col("sepal_length").gt(lit(5)))
        .group_by(vec![col("species")])
        .agg([col("sepal_width").mean()]);

    let df = q.with_streaming(true).collect()?;
    println!("{}", df);
    // --8<-- [end:streaming]

    Ok(())
}
