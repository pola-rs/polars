use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:list-example]
    // crate chrono must be added to Cargo.toml
    use chrono::{NaiveDate, NaiveDateTime};

    let names = Series::new(
        "names".into(),
        &[
            Series::new("".into(), &["Anne", "Averill", "Adams"]),
            Series::new("".into(), &["Brandon", "Brooke", "Borden", "Branson"]),
            Series::new("".into(), &["Camila", "Campbell"]),
            Series::new("".into(), &["Dennis", "Doyle"]),
        ],
    );

    let children_ages = Series::new(
        "children_ages".into(),
        &[
            Series::new("".into(), &[5_i64, 7]),
            Series::new("".into(), &[] as &[i64]),
            Series::new("".into(), &[] as &[i64]),
            Series::new("".into(), &[8_i64, 11, 18]),
        ],
    );

    let appointment_time = NaiveDate::from_ymd_opt(2022, 5, 22)
        .unwrap()
        .and_hms_micro_opt(16, 30, 0, 0)
        .unwrap();

    let medical_appointments = Series::new(
        "medical_appointments".into(),
        &[
            Series::new("".into(), &[] as &[NaiveDateTime]),
            Series::new("".into(), &[] as &[NaiveDateTime]),
            Series::new("".into(), &[] as &[NaiveDateTime]),
            Series::new("".into(), &[appointment_time]),
        ],
    )
    .cast(&DataType::List(Box::new(DataType::Datetime(
        TimeUnit::Microseconds,
        None,
    ))))?;

    let df = DataFrame::new_infer_height(vec![
        names.into(),
        children_ages.into(),
        medical_appointments.into(),
    ])?;

    println!("{df}");
    // --8<-- [end:list-example]

    // --8<-- [start:array-example]

    // 1. Construct the bit_flags array column
    let bit_flags = Series::new(
        "bit_flags".into(),
        [
            Series::new("".into(), [true, true, true, true, false]),
            Series::new("".into(), [false, true, true, true, true]),
        ],
    )
    .cast(&DataType::Array(Box::new(DataType::Boolean), 5))?;

    // 2. Construct the nested tic_tac_toe array column
    let board_1 = [[" ", "x", "o"], [" ", "x", " "], ["o", "x", " "]];
    let board_2 = [["o", "x", "x"], [" ", "o", "x"], [" ", " ", "o"]];
    let to_series = |board: &[[&str; 3]; 3]| {
        let rows: Vec<Series> = board
            .iter()
            .map(|row| Series::new("".into(), row.as_slice()))
            .collect();
        Series::new("".into(), rows)
    };

    let tic_tac_toe = Series::new(
        "tic_tac_toe".into(),
        [to_series(&board_1), to_series(&board_2)],
    )
    .cast(&DataType::Array(
        Box::new(DataType::Array(Box::new(DataType::String), 3)),
        3,
    ))?;

    // 3. Assemble and display the resulting DataFrame
    let df = DataFrame::new_infer_height(vec![bit_flags.into(), tic_tac_toe.into()])?;
    println!("{df}");
    // --8<-- [end:array-example]

    // --8<-- [start:numpy-array-inference]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:numpy-array-inference]

    // --8<-- [start:weather]
    let stations: Vec<String> = (1..=5)
        .map(|station| format!("Station {station}"))
        .collect();
    let temperatures = vec![
        "20 5 5 E1 7 13 19 9 6 20",
        "18 8 16 11 23 E2 8 E2 E2 E2 90 70 40",
        "19 24 E9 16 6 12 10 22",
        "E2 E0 15 7 8 10 E1 24 17 13 6",
        "14 8 E0 16 22 24 E1",
    ];
    let weather = df! {
        "station" => stations,
        "temperatures" => temperatures,
    }?;
    println!("{weather}");
    // --8<-- [end:weather]

    // --8<-- [start:split]
    let weather = weather
        .lazy()
        .with_columns([col("temperatures").str().split(lit(" "))])
        .collect()?;
    println!("{weather}");
    // --8<-- [end:split]

    // --8<-- [start:explode]
    let result = weather
        .clone()
        .lazy()
        .explode(
            cols(["temperatures"]),
            ExplodeOptions {
                empty_as_null: true,
                keep_nulls: true,
            },
        )
        .collect()?;
    println!("{result}");
    // --8<-- [end:explode]

    // --8<-- [start:list-slicing]
    let result = weather
        .clone()
        .lazy()
        .with_columns([
            col("temperatures").list().head(lit(3)).alias("head"),
            col("temperatures").list().tail(lit(3)).alias("tail"),
            col("temperatures")
                .list()
                .slice(lit(-3), lit(2))
                .alias("two_next_to_last"),
        ])
        .collect()?;
    println!("{result}");
    // --8<-- [end:list-slicing]

    // --8<-- [start:element-wise-casting]
    let result = weather
        .clone()
        .lazy()
        .with_columns([col("temperatures")
            .list()
            .eval(element().cast(DataType::Int64).is_null())
            .list()
            .sum()
            .alias("errors")])
        .collect()?;
    println!("{result}");
    // --8<-- [end:element-wise-casting]

    // --8<-- [start:element-wise-regex]
    // the regex feature must be enabled
    let result2 = weather
        .clone()
        .lazy()
        .with_columns([col("temperatures")
            .list()
            .eval(element().str().contains(lit("(?i)[a-z]"), true))
            .list()
            .sum()
            .alias("errors")])
        .collect()?;
    println!("{}", result.equals(&result2));
    // --8<-- [end:element-wise-regex]

    // --8<-- [start:children]
    let df = df! {
        "children" => &[
            df! {
                "name" => &["Anne", "Averill"],
                "age" => [5_i64, 7_i64],
            }?.into_struct("".into()).into_series(),
            df! {
                "name" => &["Brandon", "Brooke", "Branson"],
                "age" => &[12_i64, 9_i64, 11_i64],
            }?.into_struct("".into()).into_series(),
            df! {
                "name" => &["Camila"],
                "age" => &[19_i64],
            }?.into_struct("".into()).into_series(),
            df! {
                "name" => &["Dennis", "Doyle", "Dina"],
                "age" => &[8_i64, 11_i64, 18_i64],
            }?.into_struct("".into()).into_series(),
        ],
    }?;

    println!("{df}");
    // --8<-- [end:children]

    // --8<-- [start:list-sorting]
    let result = df
        .clone()
        .lazy()
        .select([
            col("children")
                .list()
                .eval(
                    element()
                        .sort_by(
                            [element().struct_().field_by_name("age")],
                            SortMultipleOptions::default().with_order_descending(true),
                        )
                        .struct_()
                        .field_by_name("name"),
                )
                .alias("names_by_age"),
            col("children")
                .list()
                .eval(element().struct_().field_by_name("age").min())
                .alias("min_age"),
            col("children")
                .list()
                .eval(element().struct_().field_by_name("age").max())
                .alias("max_age"),
        ])
        .collect()?;
    println!("{result}");
    // --8<-- [end:list-sorting]

    // --8<-- [start:list-aggregation]
    let result = df
        .clone()
        .lazy()
        .select([
            col("children")
                .list()
                .eval(
                    element()
                        .sort_by(
                            [element().struct_().field_by_name("age")],
                            SortMultipleOptions::default().with_order_descending(true),
                        )
                        .struct_()
                        .field_by_name("name"),
                )
                .alias("names_by_age"),
            col("children")
                .list()
                .agg(element().struct_().field_by_name("age").min())
                .alias("min_age"),
            col("children")
                .list()
                .agg(element().struct_().field_by_name("age").max())
                .alias("max_age"),
        ])
        .collect()?;
    println!("{result}");
    // --8<-- [end:list-aggregation]

    // --8<-- [start:list-entropy]
    let result = df
        .clone()
        .lazy()
        .with_columns([col("children")
            .list()
            .agg(
                element()
                    .struct_()
                    .field_by_name("age")
                    .entropy(std::f64::consts::E, true),
            )
            .alias("age_entropy")])
        .collect()?;
    println!("{result}");
    // --8<-- [end:list-entropy]

    // --8<-- [start:weather_by_day]
    let stations: Vec<String> = (1..=10).map(|idx| format!("Station {idx}")).collect();
    let weather_by_day = df! {
        "station" => stations,
        "day_1" => [17_i64, 11, 8, 22, 9, 21, 20, 8, 8, 17],
        "day_2" => [15_i64, 11, 10, 8, 7, 14, 18, 21, 15, 13],
        "day_3" => [16_i64, 15, 24, 24, 8, 23, 19, 23, 16, 10],
    }?;
    println!("{weather_by_day}");
    // --8<-- [end:weather_by_day]

    // --8<-- [start:rank_pct]
    // the feature round_series must be enabled
    let rank_pct = (element().rank(
        RankOptions {
            method: RankMethod::Average,
            descending: true,
        },
        None,
    ) / element().count())
    .round(2, RoundMode::default());

    let result = weather_by_day
        .lazy()
        .with_columns(
            [concat_list([all().exclude_cols(["station"]).as_expr()])?.alias("all_temps")],
        )
        .select([
            all().exclude_cols(["all_temps"]).as_expr(),
            col("all_temps").list().eval(rank_pct).alias("temps_rank"),
        ])
        .collect()?;

    println!("{result}");
    // --8<-- [end:rank_pct]

    // --8<-- [start:array-overview]
    // the feature is_in must be enabled
    let first_last = Series::new(
        "first_last".into(),
        &[
            Series::new("".into(), &["Anne", "Adams"]),
            Series::new("".into(), &["Brandon", "Branson"]),
            Series::new("".into(), &["Camila", "Campbell"]),
            Series::new("".into(), &["Dennis", "Doyle"]),
        ],
    )
    .cast(&DataType::Array(Box::new(DataType::String), 2))?;

    let fav_numbers = Series::new(
        "fav_numbers".into(),
        &[
            Series::new("".into(), &[42_i32, 0, 1]),
            Series::new("".into(), &[2_i32, 3, 5]),
            Series::new("".into(), &[13_i32, 21, 34]),
            Series::new("".into(), &[73_i32, 3, 7]),
        ],
    )
    .cast(&DataType::Array(Box::new(DataType::Int32), 3))?;

    let df = DataFrame::new_infer_height(vec![first_last.into(), fav_numbers.into()])?;

    let result = df
        .lazy()
        .select([
            col("first_last").arr().join(lit(" "), true).alias("name"),
            col("fav_numbers").arr().sort(SortOptions::default()),
            col("fav_numbers").arr().max().alias("largest_fav"),
            col("fav_numbers").arr().sum().alias("summed"),
            col("fav_numbers")
                .arr()
                .contains(lit(3), true)
                .alias("likes_3"),
        ])
        .collect()?;
    println!("{result}");
    // --8<-- [end:array-overview]

    Ok(())
}
