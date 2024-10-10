fn main() {
    // --8<-- [start:series]
    use polars::prelude::*;

    let s = Series::new("ints".into(), &[1, 2, 3, 4, 5]);

    println!("{}", s);
    // --8<-- [end:series]

    // --8<-- [start:series-dtype]
    let s1 = Series::new("ints".into(), &[1, 2, 3, 4, 5]);
    let s2 = Series::new("uints".into(), &[1, 2, 3, 4, 5])
        .cast(&DataType::UInt64) // Here, we actually cast after inference.
        .unwrap();
    println!("{} {}", s1.dtype(), s2.dtype()); // i32 u64
                                               // --8<-- [end:series-dtype]

    // --8<-- [start:df]
    use chrono::prelude::*;

    let df: DataFrame = df!(
        "name" => ["Alice Archer", "Ben Brown", "Chloe Cooper", "Daniel Donovan"],
        "birthdate" => [
            NaiveDate::from_ymd_opt(1997, 1, 10).unwrap(),
            NaiveDate::from_ymd_opt(1985, 2, 15).unwrap(),
            NaiveDate::from_ymd_opt(1983, 3, 22).unwrap(),
            NaiveDate::from_ymd_opt(1981, 4, 30).unwrap(),
        ],
        "weight" => [57.9, 72.5, 53.6, 83.1],  // (kg)
        "height" => [1.56, 1.77, 1.65, 1.75],  // (m)
    )
    .unwrap();
    println!("{}", df);
    // --8<-- [end:df]

    // --8<-- [start:schema]
    println!("{:?}", df.schema());
    // --8<-- [end:schema]

    // --8<-- [start:head]
    let df_head = df.head(Some(3));

    println!("{}", df_head);
    // --8<-- [end:head]

    // --8<-- [start:tail]
    let df_tail = df.tail(Some(3));

    println!("{}", df_tail);
    // --8<-- [end:tail]

    // --8<-- [start:sample]
    let n = Series::new("".into(), &[2]);
    let sampled_df = df.sample_n(&n, false, false, None).unwrap();

    println!("{}", sampled_df);
    // --8<-- [end:sample]

    // --8<-- [start:describe]
    // Not available in Rust
    // --8<-- [end:describe]
}
