fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:list-example]
    use chrono::prelude::*;
    use polars::chunked_array::builder::get_list_builder;
    use polars::prelude::*;
    let mut names = get_list_builder(&DataType::String, 4, 4, "names".into());
    names
        .append_series(&Series::new("".into(), ["Anne", "Averill", "Adams"]))
        .unwrap();
    names
        .append_series(&Series::new(
            "".into(),
            ["Brandon", "Brooke", "Borden", "Branson"],
        ))
        .unwrap();
    names
        .append_series(&Series::new("".into(), ["Camila", "Campbell"]))
        .unwrap();
    names
        .append_series(&Series::new("".into(), ["Dennis", "Doyle"]))
        .unwrap();
    let names_s = names.finish().into_series();
    let empty_i64 = Series::new_empty("".into(), &DataType::Int64);
    let mut children_ages = get_list_builder(&DataType::Int64, 4, 4, "children_ages".into());
    children_ages
        .append_series(&Series::new("".into(), [5i64, 7i64]))
        .unwrap();
    children_ages.append_series(&empty_i64).unwrap();
    children_ages.append_series(&empty_i64).unwrap();
    children_ages
        .append_series(&Series::new("".into(), [8i64, 11i64, 18i64]))
        .unwrap();
    let children_ages_s = children_ages.finish().into_series();

    let empty_dt = Series::new_empty("".into(), &DataType::Datetime(TimeUnit::Microseconds, None));
    let mut medical_appointments = get_list_builder(
        &DataType::Datetime(TimeUnit::Microseconds, None),
        4,
        4,
        "medical_appointments".into(),
    );
    medical_appointments.append_series(&empty_dt).unwrap();
    medical_appointments.append_series(&empty_dt).unwrap();
    medical_appointments.append_series(&empty_dt).unwrap();
    medical_appointments
        .append_series(&Series::new(
            "".into(),
            [NaiveDate::from_ymd_opt(2022, 5, 22)
                .unwrap()
                .and_hms_opt(16, 30, 0)
                .unwrap()
                .and_utc()
                .timestamp_micros()],
        ))
        .unwrap();
    let medical_appointments_s = medical_appointments.finish().into_series();

    let df = DataFrame::new(vec![
        names_s.into(),
        children_ages_s.into(),
        medical_appointments_s.into(),
    ])
    .unwrap();
    eprintln!("{}", df);
    // --8<-- [end:list-example]

    // --8<-- [start:array-example]
    use polars::prelude::*;
    let df = DataFrame::new(vec![
        Series::new(
            "bit_flags".into(),
            [true, true, true, true, false, false, true, true, true, true],
        )
        .reshape_array(&[
            ReshapeDimension::Infer,
            ReshapeDimension::Specified(Dimension::new(5)),
        ])
        .unwrap()
        .into(),
        Series::new(
            "tic_tac_toe".into(),
            [
                " ", "x", "o", " ", "x", " ", "o", "x", " ", "o", "x", "x", " ", "o", "x", " ",
                " ", "o",
            ],
        )
        .reshape_array(&[
            ReshapeDimension::Infer,
            ReshapeDimension::Specified(Dimension::new(3)),
            ReshapeDimension::Specified(Dimension::new(3)),
        ])
        .unwrap()
        .into(),
    ])
    .unwrap();
    eprintln!("{}", df);
    // --8<-- [end:array-example]

    // --8<-- [start:numpy-array-inference]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:numpy-array-inference]

    // --8<-- [start:weather]
    use polars::prelude::*;
    let station: Vec<String> = (1..6)
        .into_iter()
        .map(|idx| format!("Station {}", idx))
        .collect();
    let weather = DataFrame::new(vec![
        Series::new("station".into(), station).into(),
        Series::new(
            "temperature".into(),
            [
                "20 5 5 E1 7 13 19 9 6 20",
                "18 8 16 11 23 E2 8 E2 E2 E2 90 70 40",
                "19 24 E9 16 6 12 10 22",
                "E2 E0 15 7 8 10 E1 24 17 13 6",
                "14 8 E0 16 22 24 E1",
            ],
        )
        .into(),
    ])
    .unwrap();
    eprintln!("{:?}", weather);
    // --8<-- [end:weather]

    // --8<-- [start:split]
    let weather = weather
        .lazy()
        .with_columns(vec![col("temperature").str().split(lit(" "))])
        .collect()
        .unwrap();
    eprintln!("{:?}", weather);
    // --8<-- [end:split]

    // --8<-- [start:explode]
    let result = weather.explode(vec!["temperature"]).unwrap();
    eprintln!("{:?}", result);
    // --8<-- [end:explode]

    // --8<-- [start:list-slicing]
    let result = weather
        .clone()
        .lazy()
        .with_columns(vec![
            col("temperature").list().head(lit(3)).alias("head"),
            col("temperature").list().tail(lit(3)).alias("tail"),
            col("temperature")
                .list()
                .slice(lit(-3), lit(2))
                .alias("two_next_to_last"),
        ])
        .collect()
        .unwrap();
    eprintln!("{:?}", result);
    // --8<-- [end:list-slicing]

    // --8<-- [start:element-wise-casting]
    let result = weather
        .clone()
        .lazy()
        .with_columns(vec![col("temperature")
            .list()
            // needs feature "list_eval"
            .eval(col("").cast(DataType::Int64).is_null(), true)
            .list()
            .sum()
            .alias("errors")])
        .collect()
        .unwrap();
    eprintln!("{:?}", result);
    // --8<-- [end:element-wise-casting]

    // --8<-- [start:element-wise-regex]
    let result2 = weather
        .clone()
        .lazy()
        .with_columns(vec![col("temperature")
            .list()
            .eval(col("").str().contains(lit("(?i)[a-z]"), true), true)
            .list()
            .sum()
            .alias("errors")])
        .collect()
        .unwrap();

    eprintln!("{}", result.equals(&result2));
    // --8<-- [end:element-wise-regex]

    // --8<-- [start:weather_by_day]
    let station: Vec<String> = (1..11)
        .into_iter()
        .map(|idx| format!("Station {}", idx))
        .collect();
    let weather_by_day = DataFrame::new(vec![
        Series::new("station".into(), station).into(),
        Series::new("day_1".into(), [17, 11, 8, 22, 9, 21, 20, 8, 8, 17]).into(),
        Series::new("day_2".into(), [15, 11, 10, 8, 7, 14, 18, 21, 15, 13]).into(),
        Series::new("day_3".into(), [16, 15, 24, 24, 8, 23, 19, 23, 16, 10]).into(),
    ])
    .unwrap();

    eprintln!("{}", weather_by_day);
    // --8<-- [end:weather_by_day]

    // --8<-- [start:rank_pct]
    // needs feature "rank", "round_series"
    let rank_pct = (col("")
        .rank(
            RankOptions {
                descending: true,
                ..Default::default()
            },
            None,
        )
        // explicit cast here is necessary or else result is u32
        .cast(DataType::Float64)
        / col("*").count())
    .round(2);
    use polars_plan::dsl;
    let result = weather_by_day
        .lazy()
        .with_columns(vec![dsl::concat_list(vec![
            col("*").exclude(vec!["station"])
        ])
        .unwrap()
        .alias("all_temps")])
        .select(vec![
            col("*").exclude(vec!["all_temps"]),
            col("all_temps")
                .list()
                .eval(rank_pct, true)
                .alias("temps_rank"),
        ])
        .collect()
        .unwrap();
    eprintln!("{}", result);
    // --8<-- [end:rank_pct]

    // --8<-- [start:array-overview]
    let df = DataFrame::new(vec![
        Series::new(
            "first_last".into(),
            [
                "Anne", "Adams", "Brandon", "Branson", "Camila", "Campbell", "Dennis", "Doyle",
            ],
        )
        .reshape_array(&[
            ReshapeDimension::Infer,
            ReshapeDimension::Specified(Dimension::new(2)),
        ])
        .unwrap()
        .into(),
        Series::new(
            "fav_numbers".into(),
            [42, 0, 1, 2, 3, 5, 13, 21, 34, 73, 3, 7],
        )
        .reshape_array(&[
            ReshapeDimension::Infer,
            ReshapeDimension::Specified(Dimension::new(3)),
        ])
        .unwrap()
        .into(),
    ])
    .unwrap();

    let result = df
        .lazy()
        .select(vec![
            col("first_last").arr().join(lit(" "), true).alias("name"),
            col("fav_numbers").arr().sort(SortOptions::default()),
            col("fav_numbers").arr().max().alias("largest_fav"),
            col("fav_numbers").arr().sum().alias("summed"),
            col("fav_numbers").arr().contains(3).alias("likes_3"),
        ])
        .collect()
        .unwrap();
    eprintln!("{}", result);
    // --8<-- [end:array-overview]

    Ok(())
}
