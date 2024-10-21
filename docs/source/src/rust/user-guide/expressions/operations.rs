fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    use polars::prelude::*;
    use rand::{thread_rng, Rng};

    let mut arr = [0f64; 5];
    thread_rng().fill(&mut arr);

    let df = df! (
        "nrs" => &[Some(1), Some(2), Some(3), None, Some(5)],
        "names" => &["foo", "ham", "spam", "egg", "spam"],
        "random" => &arr,
        "groups" => &["A", "A", "B", "A", "B"],
    )?;

    println!("{}", &df);
    // --8<-- [end:dataframe]

    // --8<-- [start:arithmetic]
    let df_numerical = df
        .clone()
        .lazy()
        .select([
            (col("nrs") + lit(5)).alias("nrs + 5"),
            (col("nrs") - lit(5)).alias("nrs - 5"),
            (col("nrs") * col("random")).alias("nrs * random"),
            (col("nrs") / col("random")).alias("nrs / random"),
            (col("nrs").pow(lit(2))).alias("nrs ** 2"),
            (col("nrs") % lit(3)).alias("nrs % 3"),
        ])
        .collect()?;
    println!("{}", &df_numerical);
    // --8<-- [end:arithmetic]

    // --8<-- [start:operator-overloading]
    // --8<-- [end:operator-overloading]

    // --8<-- [start:comparison]
    let df_comparison = df
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
    println!("{}", &df_comparison);
    // --8<-- [end:comparison]

    // --8<-- [start:boolean]
    // --8<-- [end:boolean]

    // --8<-- [start:bitwise]
    // --8<-- [end:bitwise]

    // --8<-- [start:count]
    // --8<-- [end:count]

    // --8<-- [start:value_counts]
    // --8<-- [end:value_counts]

    // --8<-- [start:unique_counts]
    // --8<-- [end:unique_counts]

    // --8<-- [start:collatz]
    // --8<-- [end:collatz]

    Ok(())
}
