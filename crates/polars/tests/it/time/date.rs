use polars::prelude::*;

#[test]
#[cfg(feature = "concat_str")]
fn test_datetime_parse_overflow_7631() {
    use polars::datatypes::{AnyValue, TimeUnit};

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

    let expected = DataFrame::new_infer_height(vec![
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

#[test]
#[cfg(feature = "dtype-date")]
fn test_date_temporal_operations_11991() {
    use polars::prelude::*;

    let normal_date = 18628; // 2021-01-01
    let s = Int32Chunked::new("".into(), &[normal_date])
        .into_date()
        .into_series();

    let year = s.year().unwrap();
    assert_eq!(year.get(0), Some(2021));

    let month = s.month().unwrap();
    assert_eq!(month.get(0), Some(1));

    let day = s.day().unwrap();
    assert_eq!(day.get(0), Some(1));

    // Null values should remain null (regression test for #15313)
    let s_with_null = Int32Chunked::new("".into(), &[Some(18628), None])
        .into_date()
        .into_series();

    let year_with_null = s_with_null.year().unwrap();
    assert_eq!(year_with_null.get(0), Some(2021));
    assert_eq!(year_with_null.get(1), None);
}

#[test]
#[cfg(feature = "dtype-date")]
fn test_out_of_range_date_year_11991() {
    use polars::prelude::*;

    // Out-of-range dates should return null instead of panicking or returning wrong values
    // Regression test for #11991 where out-of-range dates silently returned the input value
    let out_of_range_date = -96_465_659;
    let s = Int32Chunked::new("".into(), &[out_of_range_date])
        .into_date()
        .into_series();

    let year = s.year().unwrap();
    // Should return null, not the input value -96465659
    assert_eq!(year.get(0), None);

    // is_leap_year should also return null for out-of-range dates
    let is_leap = s.is_leap_year().unwrap();
    assert_eq!(is_leap.get(0), None);
}

#[test]
#[cfg(feature = "dtype-date")]
fn test_naivedate_to_literal_29229() {
    use chrono::NaiveDate;

    fn compare(value: i64, time_unit: TimeUnit, year: i32, month: u32, day: u32) {
        let expr = Expr::Literal(Scalar::new_datetime(value, time_unit, None).into());
        let lit = NaiveDate::from_ymd_opt(year, month, day).unwrap().lit();
        assert_eq!(lit, expr);
    }

    // NaiveDate[Time] values outside of the nanosecond resolution range should not panic
    // on conversion to a literal.
    compare(-9_223_286_400_000_000_000, TimeUnit::Nanoseconds, 1677, 9, 22);
    compare(-9_223_372_800_000_000, TimeUnit::Microseconds, 1677, 9, 21);
    compare(-18_429_206_400_000_000, TimeUnit::Microseconds, 1386, 1, 1);

    compare(9_223_286_400_000_000_000, TimeUnit::Nanoseconds, 2262, 4, 11);
    compare(9_223_372_800_000_000, TimeUnit::Microseconds, 2262, 4, 12);
    compare(18_460_742_400_000_000, TimeUnit::Microseconds, 2554, 12, 31);

    compare(-8_334_601_228_800_000_000, TimeUnit::Microseconds, -262143, 1, 1);
    compare(8_210_266_790_400_000_000, TimeUnit::Microseconds, 262142, 12, 31);
}