fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:list-example]
    use chrono::prelude::*;
    use polars::prelude::*;
    let df = DataFrame::new(vec![
        Series::from_any_values(
            "names".into(),
            &[
                vec!["Anne", "Averill", "Adams"],
                vec!["Brandon", "Brooke", "Borden", "Branson"],
                vec!["Camila", "Campbell"],
                vec!["Dennis", "Doyle"],
            ]
            .iter()
            .map(|item| AnyValue::List(Series::new("".into(), item)))
            .collect::<Vec<AnyValue>>(),
            true,
        )?
        .into(),
        Series::from_any_values(
            "children_ages".into(),
            &[vec![5, 7], vec![], vec![], vec![8, 11, 18]]
                .iter()
                .map(|item| AnyValue::List(Series::new("".into(), item)))
                .collect::<Vec<AnyValue>>(),
            true,
        )?
        .into(),
        Series::from_any_values(
            "medical_appointments".into(),
            &[
                vec![],
                vec![],
                vec![],
                vec![NaiveDate::from_ymd_opt(2022, 5, 22)
                    .unwrap()
                    .and_hms_opt(16, 30, 0)
                    .unwrap()],
            ]
            .iter()
            .map(|item| AnyValue::List(Series::new("".into(), item)))
            .collect::<Vec<AnyValue>>(),
            true,
        )?
        .into(),
    ])?;
    eprintln!("{}", df);
    // --8<-- [end:list-example]

    // --8<-- [start:array-example]
    // need feature "dtype-array"
    let df = DataFrame::new(vec![
        Series::from_any_values_and_dtype(
            "bit_flags".into(),
            &[
                vec![true, true, true, true, false],
                vec![false, true, true, true, true],
            ]
            .iter()
            .map(|item| AnyValue::List(Series::new("".into(), item)))
            .collect::<Vec<AnyValue>>(),
            &DataType::Array(Box::new(DataType::Boolean), 5),
            true,
        )?
        .into(),
        Series::from_any_values_and_dtype(
            "tic_tac_toe".into(),
            &[
                vec![
                    vec![" ", "x", "o"],
                    vec![" ", "x", " "],
                    vec!["o", "x", " "],
                ],
                vec![
                    vec!["o", "x", "x"],
                    vec![" ", "o", "x"],
                    vec![" ", " ", "o"],
                ],
            ]
            .iter()
            .map(|item| {
                AnyValue::List(
                    Series::from_any_values_and_dtype(
                        "".into(),
                        &item
                            .iter()
                            .map(|ite| AnyValue::List(Series::new("".into(), ite)))
                            .collect::<Vec<AnyValue>>(),
                        &DataType::Array(Box::new(DataType::String), 3),
                        true,
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<AnyValue>>(),
            &DataType::Array(Box::new(DataType::Array(Box::new(DataType::String), 3)), 3),
            true,
        )?
        .into(),
    ])?;
    // the tic_tac_toe Series could also be defined this way, using reshape_array, instead
    let s = Series::new(
        "tic_tac_toe".into(),
        &[
            vec![
                vec![" ", "x", "o"],
                vec![" ", "x", " "],
                vec!["o", "x", " "],
            ],
            vec![
                vec!["o", "x", "x"],
                vec![" ", "o", "x"],
                vec![" ", " ", "o"],
            ],
        ]
        .into_iter()
        .flat_map(|inner| inner.into_iter().flatten())
        .collect::<Vec<&str>>(),
    )
    .reshape_array(&[
        ReshapeDimension::Infer,
        ReshapeDimension::Specified(Dimension::new(3)),
        ReshapeDimension::Specified(Dimension::new(3)),
    ])?;
    let tic_tac_toe = df.get_columns()[1].as_series().unwrap().clone();

    assert_eq!(s, tic_tac_toe);
    eprintln!("{}", df);
    // --8<-- [end:array-example]

    // --8<-- [start:numpy-array-inference]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:numpy-array-inference]

    // --8<-- [start:weather]
    let weather = DataFrame::new(vec![
        Series::new(
            "station".into(),
            &(1..6)
                .map(|idx| format!("Station {}", idx))
                .collect::<Vec<String>>(),
        )
        .into(),
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
    ])?;
    eprintln!("{}", weather);
    // --8<-- [end:weather]

    // --8<-- [start:split]
    let weather = weather
        .lazy()
        .with_columns(vec![col("temperature").str().split(lit(" "))])
        .collect()?;
    eprintln!("{}", weather);
    // --8<-- [end:split]

    // --8<-- [start:explode]
    let result = weather.explode(vec!["temperature"])?;
    eprintln!("{}", result);
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
        .collect()?;
    eprintln!("{}", result);
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
        .collect()?;
    eprintln!("{}", result);
    // --8<-- [end:element-wise-casting]

    // --8<-- [start:element-wise-regex]
    let result2 = weather
        .clone()
        .lazy()
        .with_columns(vec![col("temperature")
            .list()
            // in rust use `col("")` instead of pl.element()
            .eval(col("").str().contains(lit("(?i)[a-z]"), true), true)
            .list()
            .sum()
            .alias("errors")])
        .collect()?;
    eprintln!("{}", result.equals(&result2));
    // --8<-- [end:element-wise-regex]

    // --8<-- [start:weather_by_day]
    let weather_by_day = DataFrame::new(vec![
        Series::new(
            "station".into(),
            &(1..11)
                .map(|idx| format!("Station {}", idx))
                .collect::<Vec<String>>(),
        )
        .into(),
        Series::new("day_1".into(), [17, 11, 8, 22, 9, 21, 20, 8, 8, 17]).into(),
        Series::new("day_2".into(), [15, 11, 10, 8, 7, 14, 18, 21, 15, 13]).into(),
        Series::new("day_3".into(), [16, 15, 24, 24, 8, 23, 19, 23, 16, 10]).into(),
    ])?;
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
        // in rust use `col("*")` instead of pl.all()
        / col("*").count())
    .round(2);
    let result = weather_by_day
        .lazy()
        .with_columns(vec![
            concat_list(vec![col("*").exclude(vec!["station"])])?.alias("all_temps")
        ])
        .select(vec![
            col("*").exclude(vec!["all_temps"]),
            col("all_temps")
                .list()
                .eval(rank_pct, true)
                .alias("temps_rank"),
        ])
        .collect()?;
    eprintln!("{}", result);
    // --8<-- [end:rank_pct]

    // --8<-- [start:array-overview]
    let df = DataFrame::new(vec![
        Series::from_any_values_and_dtype(
            "first_last".into(),
            &[
                vec!["Anne", "Adams"],
                vec!["Brandon", "Branson"],
                vec!["Camila", "Campbell"],
                vec!["Dennis", "Doyle"],
            ]
            .iter()
            .map(|item| AnyValue::List(Series::new("".into(), item)))
            .collect::<Vec<AnyValue>>(),
            &DataType::Array(Box::new(DataType::String), 2),
            true,
        )?
        .into(),
        Series::from_any_values_and_dtype(
            "fav_numbers".into(),
            &[
                vec![42, 0, 1],
                vec![2, 3, 5],
                vec![13, 21, 34],
                vec![73, 3, 7],
            ]
            .iter()
            .map(|item| AnyValue::List(Series::new("".into(), item)))
            .collect::<Vec<AnyValue>>(),
            &DataType::Array(Box::new(DataType::Int32), 3),
            true,
        )?
        .into(),
    ])?;

    let result = df
        .lazy()
        .select(vec![
            col("first_last").arr().join(lit(" "), true).alias("name"),
            col("fav_numbers").arr().sort(SortOptions::default()),
            col("fav_numbers").arr().max().alias("largest_fav"),
            col("fav_numbers").arr().sum().alias("summed"),
            col("fav_numbers").arr().contains(3).alias("likes_3"),
        ])
        .collect()?;
    eprintln!("{}", result);
    // --8<-- [end:array-overview]
    Ok(())
}
