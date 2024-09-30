use chrono::prelude::*;
use polars::prelude::*;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let mut rng = rand::thread_rng();

    let df: DataFrame = df!(
        "a" => 0..6,
        "b"=> (0..6).map(|_| rng.gen::<f64>()).collect::<Vec<f64>>(),
        "c"=> [
            NaiveDate::from_ymd_opt(2025, 12, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 4).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 5).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 12, 6).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ],
        "d"=> [Some(1.0), Some(2.0), None, Some(-42.), None, Some(3.1415)],
        "e"=> ["X", "X", "Y", "X", "Z", "Y"],
    )
    .unwrap();
    // --8<-- [end:df]

    // --8<-- [start:select]
    let out = df.clone().lazy().select([col("c")]).collect()?;
    println!("{}", out);
    // --8<-- [end:select]

    // --8<-- [start:select2]
    let out = df.clone().lazy().select([col("a"), col("b")]).collect()?;
    println!("{}", out);
    // --8<-- [end:select2]

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
        .with_columns([(col("b") + lit(42)).alias("b+42")])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:with_columns]

    // --8<-- [start:group_by]
    let out = df.clone().lazy().group_by(["e"]).agg([len()]).collect()?;
    println!("{}", out);
    // --8<-- [end:group_by]

    // --8<-- [start:group_by2]
    let out = df
        .clone()
        .lazy()
        .group_by(["e"])
        .agg([col("a").max().alias("max_a"), col("b").sum().alias("sum_b")])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:group_by2]

    // --8<-- [start:complex]
    let start_date = NaiveDate::from_ymd_opt(2025, 12, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let end_date = NaiveDate::from_ymd_opt(2025, 12, 5)
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
        .group_by(["e"])
        .agg([
            len().alias("count"),
            col("a").max().alias("max_a"),
            col("b").sum().alias("sum_b"),
        ])
        .with_columns([(col("max_a") * col("sum_b")).alias("times")])
        .select([col("*").exclude(["max_a", "sum_b"])])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:complex]
    // --8<-- [start:csv]
    // --8<-- [end:csv]
    // --8<-- [start:expression-expansion]
    // --8<-- [end:expression-expansion]
    // --8<-- [start:filter-multiple]
    // --8<-- [end:filter-multiple]
    // --8<-- [start:group_by-agg]
    // --8<-- [end:group_by-agg]
    // --8<-- [start:join]
    // --8<-- [end:join]
    // --8<-- [start:concat]
    // --8<-- [end:concat]

    Ok(())
}
