use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:lazy]
    let q = LazyCsvReader::new(PlRefPath::new("docs/assets/data/iris.csv"))
        .with_has_header(true)
        .finish()?
        .filter(col("sepal_length").gt(lit(5)))
        .group_by(vec![col("species")])
        .agg([col("sepal_width").mean()]);
    let df = q.collect()?;
    println!("{df}");
    // --8<-- [end:lazy]

    // --8<-- [start:explain]
    let q = LazyCsvReader::new(PlRefPath::new("docs/assets/data/iris.csv"))
        .with_has_header(true)
        .finish()?
        .filter(col("sepal_length").gt(lit(5)))
        .group_by(vec![col("species")])
        .agg([col("sepal_width").mean()]);
    println!("{}", q.explain(true)?);
    // --8<-- [end:explain]

    Ok(())
}
