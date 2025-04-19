fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    use polars::prelude::*;

    let df = df! (
        "nrs" => &[Some(1), Some(2), Some(3), None, Some(5)],
        "names" => &["foo", "ham", "spam", "egg", "spam"],
        "random" => &[0.37454, 0.950714, 0.731994, 0.598658, 0.156019],
        "groups" => &["A", "A", "B", "A", "B"],
    )?;

    println!("{}", &df);
    // --8<-- [end:dataframe]

    // --8<-- [start:arithmetic]
    let result = df
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
    println!("{}", result);
    // --8<-- [end:arithmetic]

    // --8<-- [start:comparison]
    let result = df
        .clone()
        .lazy()
        .select([
            col("nrs").gt(1).alias("nrs > 1"),
            col("nrs").gt_eq(3).alias("nrs >= 3"),
            col("random").lt_eq(0.2).alias("random < .2"),
            col("random").lt_eq(0.5).alias("random <= .5"),
            col("nrs").neq(1).alias("nrs != 1"),
            col("nrs").eq(1).alias("nrs == 1"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:comparison]

    // --8<-- [start:boolean]
    let result = df
        .clone()
        .lazy()
        .select([
            ((col("nrs").is_null()).not().and(col("groups").eq(lit("A"))))
                .alias("number not null and group A"),
            (col("random").lt(lit(0.5)).or(col("groups").eq(lit("B"))))
                .alias("random < 0.5 or group B"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:boolean]

    // --8<-- [start:bitwise]
    let result = df
        .clone()
        .lazy()
        .select([
            col("nrs"),
            col("nrs").and(lit(6)).alias("nrs & 6"),
            col("nrs").or(lit(6)).alias("nrs | 6"),
            col("nrs").not().alias("not nrs"),
            col("nrs").xor(lit(6)).alias("nrs ^ 6"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:bitwise]

    // --8<-- [start:count]
    use rand::distributions::{Distribution, Uniform};
    use rand::thread_rng;

    let mut rng = thread_rng();
    let between = Uniform::new_inclusive(0, 100_000);
    let arr: Vec<u32> = between.sample_iter(&mut rng).take(100_100).collect();

    let long_df = df!(
        "numbers" => &arr
    )?;

    let result = long_df
        .clone()
        .lazy()
        .select([
            col("numbers").n_unique().alias("n_unique"),
            col("numbers").approx_n_unique().alias("approx_n_unique"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:count]

    // --8<-- [start:value_counts]
    let result = df
        .clone()
        .lazy()
        .select([col("names")
            .value_counts(false, false, "count", false)
            .alias("value_counts")])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:value_counts]

    // --8<-- [start:unique_counts]
    let result = df
        .clone()
        .lazy()
        .select([
            col("names").unique_stable().alias("unique"),
            col("names").unique_counts().alias("unique_counts"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:unique_counts]

    // --8<-- [start:collatz]
    let result = df
        .clone()
        .lazy()
        .select([
            col("nrs"),
            when((col("nrs") % lit(2)).eq(lit(1)))
                .then(lit(3) * col("nrs") + lit(1))
                .otherwise(col("nrs") / lit(2))
                .alias("Collatz"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:collatz]

    Ok(())
}
