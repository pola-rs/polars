use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    use rand::{thread_rng, Rng};

    let mut arr = [0f64; 5];
    thread_rng().fill(&mut arr);

    let df = df! (
        "nrs" => &[Some(1), Some(2), Some(3), None, Some(5)],
        "names" => &["foo", "ham", "spam", "egg", "spam"],
        "random" => &arr,
        "groups" => &["A", "A", "B", "C", "B"],
    )?;

    println!("{}", &df);
    // --8<-- [end:dataframe]

    // --8<-- [start:samename]
    let df_samename = df.clone().lazy().select([col("nrs") + lit(5)]).collect()?;
    println!("{}", &df_samename);
    // --8<-- [end:samename]

    // --8<-- [start:samenametwice]
    let df_samename2 = df
        .clone()
        .lazy()
        .select([col("nrs") + lit(5), col("nrs") - lit(5)])
        .collect();
    match df_samename2 {
        Ok(df) => println!("{}", &df),
        Err(e) => println!("{:?}", &e),
    };
    // --8<-- [end:samenametwice]

    // --8<-- [start:samenamealias]
    let df_alias = df
        .clone()
        .lazy()
        .select([
            (col("nrs") + lit(5)).alias("nrs + 5"),
            (col("nrs") - lit(5)).alias("nrs - 5"),
        ])
        .collect()?;
    println!("{}", &df_alias);
    // --8<-- [end:samenamealias]

    // --8<-- [start:countunique]
    let df_alias = df
        .clone()
        .lazy()
        .select([
            col("names").n_unique().alias("unique"),
            // Following query shows there isn't anything in Rust API
            // https://docs.rs/polars/latest/polars/?search=approx_n_unique
            // col("names").approx_n_unique().alias("unique_approx"),
        ])
        .collect()?;
    println!("{}", &df_alias);
    // --8<-- [end:countunique]

    // --8<-- [start:conditional]
    let df_conditional = df
        .clone()
        .lazy()
        .select([
            col("nrs"),
            when(col("nrs").gt(2))
                .then(lit(true))
                .otherwise(lit(false))
                .alias("conditional"),
        ])
        .collect()?;
    println!("{}", &df_conditional);
    // --8<-- [end:conditional]

    Ok(())
}
