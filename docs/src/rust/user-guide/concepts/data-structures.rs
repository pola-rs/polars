fn main() {
    // --8<-- [start:series]
    use polars::prelude::*;

    let s = Series::new("a", &[1, 2, 3, 4, 5]);

    println!("{}", s);
    // --8<-- [end:series]

    // --8<-- [start:dataframe]
    use chrono::NaiveDate;

    let df: DataFrame = df!(
        "integer" => &[1, 2, 3, 4, 5],
        "date" => &[
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 1, 4).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2025, 1, 5).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ],
        "float" => &[4.0, 5.0, 6.0, 7.0, 8.0]
    )
    .unwrap();

    println!("{}", df);
    // --8<-- [end:dataframe]

    // --8<-- [start:head]
    let df_head = df.head(Some(3));

    println!("{}", df_head);
    // --8<-- [end:head]

    // --8<-- [start:tail]
    let df_tail = df.tail(Some(3));

    println!("{}", df_tail);
    // --8<-- [end:tail]

    // --8<-- [start:sample]
    let n = Series::new("", &[2]);
    let sampled_df = df.sample_n(&n, false, false, None).unwrap();

    println!("{}", sampled_df);
    // --8<-- [end:sample]

    // --8<-- [start:describe]
    // Not available in Rust
    // --8<-- [end:describe]
}
