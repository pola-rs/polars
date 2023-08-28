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

// --8<-- [start:select]
    let out = df
        .clone()
        .lazy()
        .select([
            sum("nrs"),
            col("names").sort(false),
            col("names").first().alias("first name"),
            (mean("nrs") * lit(10)).alias("10xnrs"),
        ])
        .collect()?;
    println!("{}", out);
// --8<-- [end:select]

// --8<-- [start:filter]
    let out = df.clone().lazy().filter(col("nrs").gt(lit(2))).collect()?;
    println!("{}", out);
// --8<-- [end:filter]

// --8<-- [start:with_columns]
    let out = df
        .clone()
        .lazy()
        .with_columns([
            sum("nrs").alias("nrs_sum"),
            col("random").count().alias("count"),
        ])
        .collect()?;
    println!("{}", out);
// --8<-- [end:with_columns]

    // --8<-- [start:groupby]
    let out = df
        .lazy()
        .groupby([col("groups")])
        .agg([
            sum("nrs"),                           // sum nrs by groups
            col("random").count().alias("count"), // count group members
            // sum random where name != null
            col("random")
                .filter(col("names").is_not_null())
                .sum()
                .suffix("_sum"),
            col("names").reverse().alias("reversed names"),
        ])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:groupby]
    Ok(())
}