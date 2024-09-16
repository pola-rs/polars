use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    use rand::{thread_rng, Rng};

    let mut arr = [0f64; 5];
    thread_rng().fill(&mut arr);

    let df = df! (
        "nrs" => &[Some(1), Some(2), Some(3), None, Some(5)],
        "names" => &[Some("foo"), Some("ham"), Some("spam"), Some("eggs"), None],
        "random" => &arr,
        "groups" => &["A", "A", "B", "C", "B"],
    )?;

    println!("{}", &df);
    // --8<-- [end:dataframe]

    // --8<-- [start:numerical]
    let df_numerical = df
        .clone()
        .lazy()
        .select([
            (col("nrs") + lit(5)).alias("nrs + 5"),
            (col("nrs") - lit(5)).alias("nrs - 5"),
            (col("nrs") * col("random")).alias("nrs * random"),
            (col("nrs") / col("random")).alias("nrs / random"),
        ])
        .collect()?;
    println!("{}", &df_numerical);
    // --8<-- [end:numerical]

    // --8<-- [start:logical]
    let df_logical = df
        .clone()
        .lazy()
        .select([
            col("nrs").gt(1).alias("nrs > 1"),
            col("random").lt_eq(0.5).alias("random < .5"),
            col("nrs").neq(1).alias("nrs != 1"),
            col("nrs").eq(1).alias("nrs == 1"),
            (col("random").lt_eq(0.5))
                .and(col("nrs").gt(1))
                .alias("and_expr"), // and
            (col("random").lt_eq(0.5))
                .or(col("nrs").gt(1))
                .alias("or_expr"), // or
        ])
        .collect()?;
    println!("{}", &df_logical);
    // --8<-- [end:logical]
    Ok(())
}
