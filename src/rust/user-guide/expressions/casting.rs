// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dfnum]
    let df = df! (
        "integers"=> &[1, 2, 3, 4, 5],
        "big_integers"=> &[1, 10000002, 3, 10000004, 10000005],
        "floats"=> &[4.0, 5.0, 6.0, 7.0, 8.0],
        "floats_with_decimal"=> &[4.532, 5.5, 6.5, 7.5, 8.5],
    )?;

    println!("{}", &df);
    // --8<-- [end:dfnum]

    // --8<-- [start:castnum]
    let out = df
        .clone()
        .lazy()
        .select([
            col("integers")
                .cast(DataType::Float32)
                .alias("integers_as_floats"),
            col("floats")
                .cast(DataType::Int32)
                .alias("floats_as_integers"),
            col("floats_with_decimal")
                .cast(DataType::Int32)
                .alias("floats_with_decimal_as_integers"),
        ])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:castnum]

    // --8<-- [start:downcast]
    let out = df
        .clone()
        .lazy()
        .select([
            col("integers")
                .cast(DataType::Int16)
                .alias("integers_smallfootprint"),
            col("floats")
                .cast(DataType::Float32)
                .alias("floats_smallfootprint"),
        ])
        .collect();
    match out {
        Ok(out) => println!("{}", &out),
        Err(e) => println!("{:?}", e),
    };
    // --8<-- [end:downcast]

    // --8<-- [start:overflow]

    let out = df
        .clone()
        .lazy()
        .select([col("big_integers").strict_cast(DataType::Int8)])
        .collect();
    match out {
        Ok(out) => println!("{}", &out),
        Err(e) => println!("{:?}", e),
    };
    // --8<-- [end:overflow]

    // --8<-- [start:overflow2]
    let out = df
        .clone()
        .lazy()
        .select([col("big_integers").cast(DataType::Int8)])
        .collect();
    match out {
        Ok(out) => println!("{}", &out),
        Err(e) => println!("{:?}", e),
    };
    // --8<-- [end:overflow2]

    // --8<-- [start:strings]

    let df = df! (
            "integers" => &[1, 2, 3, 4, 5],
            "float" => &[4.0, 5.03, 6.0, 7.0, 8.0],
            "floats_as_string" => &["4.0", "5.0", "6.0", "7.0", "8.0"],
    )?;

    let out = df
        .clone()
        .lazy()
        .select([
            col("integers").cast(DataType::String),
            col("float").cast(DataType::String),
            col("floats_as_string").cast(DataType::Float64),
        ])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:strings]

    // --8<-- [start:strings2]

    let df = df! ("strings_not_float"=> ["4.0", "not_a_number", "6.0", "7.0", "8.0"])?;

    let out = df
        .clone()
        .lazy()
        .select([col("strings_not_float").cast(DataType::Float64)])
        .collect();
    match out {
        Ok(out) => println!("{}", &out),
        Err(e) => println!("{:?}", e),
    };
    // --8<-- [end:strings2]

    // --8<-- [start:bool]

    let df = df! (
            "integers"=> &[-1, 0, 2, 3, 4],
            "floats"=> &[0.0, 1.0, 2.0, 3.0, 4.0],
            "bools"=> &[true, false, true, false, true],
    )?;

    let out = df
        .clone()
        .lazy()
        .select([
            col("integers").cast(DataType::Boolean),
            col("floats").cast(DataType::Boolean),
        ])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:bool]

    // --8<-- [start:dates]
    use chrono::prelude::*;

    let date = polars::time::date_range(
        "date",
        NaiveDate::from_ymd_opt(2022, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        NaiveDate::from_ymd_opt(2022, 1, 5)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        Duration::parse("1d"),
        ClosedWindow::Both,
        TimeUnit::Milliseconds,
        None,
    )?
    .cast(&DataType::Date)?;

    let datetime = polars::time::date_range(
        "datetime",
        NaiveDate::from_ymd_opt(2022, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        NaiveDate::from_ymd_opt(2022, 1, 5)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        Duration::parse("1d"),
        ClosedWindow::Both,
        TimeUnit::Milliseconds,
        None,
    )?;

    let df = df! (
        "date" => date,
        "datetime" => datetime,
    )?;

    let out = df
        .clone()
        .lazy()
        .select([
            col("date").cast(DataType::Int64),
            col("datetime").cast(DataType::Int64),
        ])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:dates]

    // --8<-- [start:dates2]
    let date = polars::time::date_range(
        "date",
        NaiveDate::from_ymd_opt(2022, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        NaiveDate::from_ymd_opt(2022, 1, 5)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        Duration::parse("1d"),
        ClosedWindow::Both,
        TimeUnit::Milliseconds,
        None,
    )?;

    let df = df! (
            "date" => date,
            "string" => &[
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
            ],
    )?;

    let out = df
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
    println!("{}", &out);
    // --8<-- [end:dates2]

    Ok(())
}
