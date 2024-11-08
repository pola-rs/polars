fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dfnum]
    use polars::prelude::*;

    let df = df! (
        "integers"=> [1, 2, 3],
        "big_integers"=> [10000002, 2, 30000003],
        "floats"=> [4.0, 5.8, -6.3],
    )?;

    println!("{}", df);
    // --8<-- [end:dfnum]

    // --8<-- [start:castnum]
    let result = df
        .clone()
        .lazy()
        .select([
            col("integers")
                .cast(DataType::Float32)
                .alias("integers_as_floats"),
            col("floats")
                .cast(DataType::Int32)
                .alias("floats_as_integers"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:castnum]

    // --8<-- [start:downcast]
    println!("Before downcasting: {} bytes", df.estimated_size());
    let result = df
        .clone()
        .lazy()
        .with_columns([
            col("integers").cast(DataType::Int16),
            col("floats").cast(DataType::Float32),
        ])
        .collect()?;
    println!("After downcasting: {} bytes", result.estimated_size());
    // --8<-- [end:downcast]

    // --8<-- [start:overflow]
    let result = df
        .clone()
        .lazy()
        .select([col("big_integers").strict_cast(DataType::Int8)])
        .collect();
    if let Err(e) = result {
        println!("{}", e)
    };
    // --8<-- [end:overflow]

    // --8<-- [start:overflow2]
    let result = df
        .clone()
        .lazy()
        .select([col("big_integers").cast(DataType::Int8)])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:overflow2]

    // --8<-- [start:strings]
    let df = df! (
        "integers_as_strings" => ["1", "2", "3"],
        "floats_as_strings" => ["4.0", "5.8", "-6.3"],
        "floats" => [4.0, 5.8, -6.3],
    )?;

    let result = df
        .clone()
        .lazy()
        .select([
            col("integers_as_strings").cast(DataType::Int32),
            col("floats_as_strings").cast(DataType::Float64),
            col("floats").cast(DataType::String),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:strings]

    // --8<-- [start:strings2]
    let df = df! ("floats" => ["4.0", "5.8", "- 6 . 3"])?;

    let result = df
        .clone()
        .lazy()
        .select([col("floats").strict_cast(DataType::Float64)])
        .collect();
    if let Err(e) = result {
        println!("{}", e)
    };
    // --8<-- [end:strings2]

    // --8<-- [start:bool]
    let df = df! (
            "integers"=> [-1, 0, 2, 3, 4],
            "floats"=> [0.0, 1.0, 2.0, 3.0, 4.0],
            "bools"=> [true, false, true, false, true],
    )?;

    let result = df
        .clone()
        .lazy()
        .select([
            col("integers").cast(DataType::Boolean),
            col("floats").cast(DataType::Boolean),
            col("bools").cast(DataType::UInt8),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:bool]

    // --8<-- [start:dates]
    use chrono::prelude::*;

    let df = df!(
        "date" => [
            NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),  // epoch
            NaiveDate::from_ymd_opt(1970, 1, 10).unwrap(),  // 9 days later
        ],
        "datetime" => [
            NaiveDate::from_ymd_opt(1970, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),  // epoch
            NaiveDate::from_ymd_opt(1970, 1, 1).unwrap().and_hms_opt(0, 1, 0).unwrap(),  // 1 minute later
        ],
        "time" => [
            NaiveTime::from_hms_opt(0, 0, 0).unwrap(),  // reference time
            NaiveTime::from_hms_opt(0, 0, 1).unwrap(),  // 1 second later
        ]
    )
    .unwrap()
    .lazy()
    // Make the time unit match that of Python's for the same results.
    .with_column(col("datetime").cast(DataType::Datetime(TimeUnit::Microseconds, None)))
    .collect()?;

    let result = df
        .clone()
        .lazy()
        .select([
            col("date").cast(DataType::Int64).alias("days_since_epoch"),
            col("datetime")
                .cast(DataType::Int64)
                .alias("us_since_epoch"),
            col("time").cast(DataType::Int64).alias("ns_since_midnight"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:dates]

    // --8<-- [start:dates2]
    let df = df! (
            "date" => [
                NaiveDate::from_ymd_opt(2022, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2022, 1, 2).unwrap(),
            ],
            "string" => [
                "2022-01-01",
                "2022-01-02",
            ],
    )?;

    let result = df
        .clone()
        .lazy()
        .select([
            col("date").dt().to_string("%Y-%m-%d"),
            col("string").str().to_datetime(
                Some(TimeUnit::Microseconds),
                None,
                StrptimeOptions::default(),
                lit("raise"),
            ),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:dates2]

    Ok(())
}
