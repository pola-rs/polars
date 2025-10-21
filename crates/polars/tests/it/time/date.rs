use polars::prelude::*;

#[test]
fn test_datetime_parse_overflow_7631() {
    let df = df![
        "year"=> &[2020, 2021, 2022],
        "month"=> &[1, 2, 3],
        "day"=> &[1, 2, 3],
    ]
    .unwrap()
    .lazy();

    let df = df.with_column(
        concat_str([col("year"), col("month"), col("day")], "-", false) // produces e.g., `2020-1-1`
            .str()
            .strptime(
                DataType::Datetime(TimeUnit::Milliseconds, None),
                StrptimeOptions {
                    format: Some("%Y-%m-%_d".into()),
                    ..Default::default()
                },
                lit("latest"),
            )
            .alias("dt1"),
    );
    let actual = df.collect().unwrap();

    let expected = DataFrame::new(vec![
        Column::new("year".into(), &[2020, 2021, 2022]),
        Column::new("month".into(), &[1, 2, 3]),
        Column::new("day".into(), &[1, 2, 3]),
        Column::new(
            "dt1".into(),
            &[
                AnyValue::Datetime(1577836800000, TimeUnit::Milliseconds, None),
                AnyValue::Datetime(1612224000000, TimeUnit::Milliseconds, None),
                AnyValue::Datetime(1646265600000, TimeUnit::Milliseconds, None),
            ],
        ),
    ])
    .unwrap();

    assert_eq!(actual, expected);
}
