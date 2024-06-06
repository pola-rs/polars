// --8<-- [start:setup]
use polars::io::prelude::*;
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_try_parse_dates(true))
        .try_into_reader_with_file_path(Some("docs/data/apple_stock.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    println!("{}", &df);
    // --8<-- [end:df]

    // --8<-- [start:cast]
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_try_parse_dates(false))
        .try_into_reader_with_file_path(Some("docs/data/apple_stock.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let df = df
        .clone()
        .lazy()
        .with_columns([col("Date").str().to_date(StrptimeOptions::default())])
        .collect()?;
    println!("{}", &df);
    // --8<-- [end:cast]

    // --8<-- [start:df3]
    let df_with_year = df
        .clone()
        .lazy()
        .with_columns([col("Date").dt().year().alias("year")])
        .collect()?;
    println!("{}", &df_with_year);
    // --8<-- [end:df3]

    // --8<-- [start:extract]
    let df_with_year = df
        .clone()
        .lazy()
        .with_columns([col("Date").dt().year().alias("year")])
        .collect()?;
    println!("{}", &df_with_year);
    // --8<-- [end:extract]

    // --8<-- [start:mixed]
    let data = [
        "2021-03-27T00:00:00+0100",
        "2021-03-28T00:00:00+0100",
        "2021-03-29T00:00:00+0200",
        "2021-03-30T00:00:00+0200",
    ];
    let q = col("date")
        .str()
        .to_datetime(
            Some(TimeUnit::Microseconds),
            None,
            StrptimeOptions {
                format: Some("%Y-%m-%dT%H:%M:%S%z".to_string()),
                ..Default::default()
            },
            lit("raise"),
        )
        .dt()
        .convert_time_zone("Europe/Brussels".to_string());
    let mixed_parsed = df!("date" => &data)?.lazy().select([q]).collect()?;

    println!("{}", &mixed_parsed);
    // --8<-- [end:mixed]

    Ok(())
}
