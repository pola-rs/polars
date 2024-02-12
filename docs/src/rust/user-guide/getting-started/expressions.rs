use chrono::prelude::*;
use polars::prelude::*;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();

    let df: DataFrame = df!(
        "a" => 0..5,
        "b"=> (0..5).map(|_| rng.gen::<f64>()).collect::<Vec<f64>>(),
        "c"=> [
            NaiveDate::from_ymd_opt(2025, 12, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 4).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 5).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ],
        "d"=> [Some(1.0), Some(2.0), None, Some(-42.), None]
    )
    .unwrap();

    // --8<-- [start:select]
    let out = df.clone().lazy().select([col("*")]).collect()?;
    println!("{}", out);
    // --8<-- [end:select]

    // --8<-- [start:select2]
    let out = df.clone().lazy().select([col("a"), col("b")]).collect()?;
    println!("{}", out);
    // --8<-- [end:select2]

    // --8<-- [start:select3]
    let out = df
        .clone()
        .lazy()
        .select([col("a"), col("b")])
        .limit(3)
        .collect()?;
    println!("{}", out);
    // --8<-- [end:select3]

    // --8<-- [start:exclude]
    let out = df
        .clone()
        .lazy()
        .select([col("*").exclude(["a", "c"])])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:exclude]

    // --8<-- [start:filter]
    let start_date = NaiveDate::from_ymd_opt(2025, 12, 2)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let end_date = NaiveDate::from_ymd_opt(2025, 12, 3)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let out = df
        .clone()
        .lazy()
        .filter(
            col("c")
                .gt_eq(lit(start_date))
                .and(col("c").lt_eq(lit(end_date))),
        )
        .collect()?;
    println!("{}", out);
    // --8<-- [end:filter]

    // --8<-- [start:filter2]
    let out = df
        .clone()
        .lazy()
        .filter(col("a").lt_eq(3).and(col("d").is_not_null()))
        .collect()?;
    println!("{}", out);
    // --8<-- [end:filter2]

    // --8<-- [start:with_columns]
    let out = df
        .clone()
        .lazy()
        .with_columns([
            col("b").sum().alias("e"),
            (col("b") + lit(42)).alias("b+42"),
        ])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:with_columns]

    // --8<-- [start:dataframe2]
    let df2: DataFrame = df!("x" => 0..8,
        "y"=> &["A", "A", "A", "B", "B", "C", "X", "X"],
    )
    .expect("should not fail");
    println!("{}", df2);
    // --8<-- [end:dataframe2]

    // --8<-- [start:group_by]
    let out = df2.clone().lazy().group_by(["y"]).agg([len()]).collect()?;
    println!("{}", out);
    // --8<-- [end:group_by]

    // --8<-- [start:group_by2]
    let out = df2
        .clone()
        .lazy()
        .group_by(["y"])
        .agg([col("*").count().alias("count"), col("*").sum().alias("sum")])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:group_by2]

    // --8<-- [start:combine]
    let out = df
        .clone()
        .lazy()
        .with_columns([(col("a") * col("b")).alias("a * b")])
        .select([col("*").exclude(["c", "d"])])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:combine]

    // --8<-- [start:combine2]
    let out = df
        .clone()
        .lazy()
        .with_columns([(col("a") * col("b")).alias("a * b")])
        .select([col("*").exclude(["d"])])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:combine2]

    Ok(())
}
