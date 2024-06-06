use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:streaming]
    let q1 = LazyCsvReader::new("docs/data/iris.csv")
        .with_has_header(true)
        .finish()?
        .filter(col("sepal_length").gt(lit(5)))
        .group_by(vec![col("species")])
        .agg([col("sepal_width").mean()]);

    let df = q1.clone().with_streaming(true).collect()?;
    println!("{}", df);
    // --8<-- [end:streaming]

    // --8<-- [start:example]
    let query_plan = q1.with_streaming(true).explain(true)?;
    println!("{}", query_plan);
    // --8<-- [end:example]

    // --8<-- [start:example2]
    let q2 = LazyCsvReader::new("docs/data/iris.csv")
        .finish()?
        .with_columns(vec![col("sepal_length")
            .mean()
            .over(vec![col("species")])
            .alias("sepal_length_mean")]);

    let query_plan = q2.with_streaming(true).explain(true)?;
    println!("{}", query_plan);
    // --8<-- [end:example2]

    Ok(())
}
