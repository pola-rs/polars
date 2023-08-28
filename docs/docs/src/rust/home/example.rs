fn main() -> Result<(), Box<dyn std::error::Error>>{

    // --8<-- [start:example]
    use polars::prelude::*;

    let q = LazyCsvReader::new("docs/src/data/iris.csv")
        .has_header(true)
        .finish()?
        .filter(col("sepal_length").gt(lit(5)))
        .groupby(vec![col("species")])
        .agg([col("*").sum()]);
    
    let df = q.collect();
    // --8<-- [end:example]

    Ok(())
}
