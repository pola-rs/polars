fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:setup]
    use std::fs::File;

    use chrono::prelude::*;
    use polars::prelude::*;

    let mut df: DataFrame = df!(
        "date" => &[
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                NaiveDate::from_ymd_opt(2025, 1, 2).unwrap().and_hms_opt(0, 0, 0).unwrap(),
                NaiveDate::from_ymd_opt(2025, 1, 3).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        ],
        "float" => &[4.0, 5.0, 6.0],
    )
    .unwrap();
    println!("{}", df);
    // --8<-- [end:setup]

    // --8<-- [start:csv]
    let mut file = File::create("docs/assets/data/output.csv").expect("could not create file");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df)?;
    let df_csv = CsvReadOptions::default()
        .with_infer_schema_length(None)
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("docs/assets/data/output.csv".into()))?
        .finish()?;
    println!("{}", df_csv);
    // --8<-- [end:csv]

    Ok(())
}
