fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:list-example]
    use chrono::prelude::*;
    use polars::prelude::*;
    use polars::chunked_array::builder::get_list_builder;
    let mut names=get_list_builder(
        &DataType::String,
        4,
        4,
        "names".into()
    );
    let _ = names.append_series(&Series::new("".into(), ["Anne", "Averill", "Adams"])).unwrap();
    let _ = names.append_series(&Series::new("".into(), ["Brandon", "Brooke", "Borden", "Branson"])).unwrap();
    let _ = names.append_series(&Series::new("".into(), ["Camila", "Campbell"])).unwrap();
    let _ = names.append_series(&Series::new("".into(), ["Dennis", "Doyle"])).unwrap();
    let names_s = names.finish().into_series();


    let mut children_ages=get_list_builder(
        &DataType::Int64, 
        4, 4, "children_ages".into());
        let _ = children_ages.append_series(&Series::new("".into(), [5i64, 7i64])).unwrap();
        let _ = children_ages.append_series(&Series::new_empty("".into(), &DataType::Int64)).unwrap();
        let _ = children_ages.append_series(&Series::new_empty("".into(), &DataType::Int64)).unwrap();
        let _ = children_ages.append_series(&Series::new("".into(), [8i64, 11i64, 18i64])).unwrap();
    let children_ages_s = children_ages.finish().into_series();


    let mut medical_appointments=get_list_builder(
        &DataType::Datetime(TimeUnit::Microseconds, None),
        4,
        4,
        "medical_appointments".into()
    );
    let _ = medical_appointments.append_series(&Series::new_empty("".into(), &DataType::Datetime(TimeUnit::Microseconds, None))).unwrap();
    let _ = medical_appointments.append_series(&Series::new_empty("".into(), &DataType::Datetime(TimeUnit::Microseconds, None))).unwrap();
    let _ = medical_appointments.append_series(&Series::new_empty("".into(), &DataType::Datetime(TimeUnit::Microseconds, None))).unwrap();
    let _ = medical_appointments.append_series(&Series::new("".into(), [NaiveDate::from_ymd_opt(2022, 5, 22).unwrap().and_hms_opt(16, 30, 0).unwrap().and_utc().timestamp_micros()])).unwrap();
    let medical_appointments_s = medical_appointments.finish().into_series();
    
    let df = DataFrame::new(vec![
        names_s.into(),
        children_ages_s.into(),
        medical_appointments_s.into()
    ]).unwrap();
    eprintln!("{}", df);
    // --8<-- [end:list-example]

    // --8<-- [start:array-example]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:array-example]

    // --8<-- [start:numpy-array-inference]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:numpy-array-inference]

    // --8<-- [start:weather]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:weather]

    // --8<-- [start:split]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:split]

    // --8<-- [start:explode]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:explode]

    // --8<-- [start:list-slicing]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:list-slicing]

    // --8<-- [start:element-wise-casting]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:element-wise-casting]

    // --8<-- [start:element-wise-regex]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:element-wise-regex]

    // --8<-- [start:weather_by_day]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:weather_by_day]

    // --8<-- [start:rank_pct]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:rank_pct]

    // --8<-- [start:array-overview]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:array-overview]

    Ok(())
}
