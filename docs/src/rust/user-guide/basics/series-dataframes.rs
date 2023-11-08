fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:series]
    use polars::prelude::*;

    let s = Series::new("a", [1, 2, 3, 4, 5]);
    println!("{}", s);
    // --8<-- [end:series]

    // --8<-- [start:minmax]
    let s = Series::new("a", [1, 2, 3, 4, 5]);
    // The use of generics is necessary for the type system
    println!("{}", s.min::<u64>().unwrap());
    println!("{}", s.max::<u64>().unwrap());
    // --8<-- [end:minmax]

    // --8<-- [start:string]
    // This operation is not directly available on the Series object yet, only on the DataFrame
    // --8<-- [end:string]

    // --8<-- [start:dt]
    // This operation is not directly available on the Series object yet, only as an Expression
    // --8<-- [end:dt]

    // --8<-- [start:dataframe]
    use chrono::prelude::*;

    let df: DataFrame = df!(
        "integer" => &[1, 2, 3, 4, 5],
        "date" => &[
            NaiveDate::from_ymd_opt(2022, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2022, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2022, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2022, 1, 4).unwrap().and_hms_opt(0, 0, 0).unwrap(),
            NaiveDate::from_ymd_opt(2022, 1, 5).unwrap().and_hms_opt(0, 0, 0).unwrap()
        ],
        "float" => &[4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();

    println!("{}", df);
    // --8<-- [end:dataframe]

    // --8<-- [start:head]
    println!("{}", df.head(Some(3)));
    // --8<-- [end:head]

    // --8<-- [start:tail]
    println!("{}", df.tail(Some(3)));
    // --8<-- [end:tail]

    // --8<-- [start:sample]
    let n = Series::new("n", [2]);
    println!("{}", df.sample_n(&n, false, true, None)?);
    // --8<-- [end:sample]

    // --8<-- [start:describe]
    println!("{:?}", df.describe(None));
    // --8<-- [end:describe]
    Ok(())
}
