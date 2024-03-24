// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:example]
    let ts = ["2021-03-27 03:00", "2021-03-28 03:00"];
    let tz_naive = Series::new("tz_naive", &ts);
    let time_zones_df = DataFrame::new(vec![tz_naive])?
        .lazy()
        .select([col("tz_naive").str().to_datetime(
            Some(TimeUnit::Milliseconds),
            None,
            StrptimeOptions::default(),
            lit("raise"),
        )])
        .with_columns([col("tz_naive")
            .dt()
            .replace_time_zone(Some("UTC".to_string()), lit("raise"), NonExistent::Raise)
            .alias("tz_aware")])
        .collect()?;

    println!("{}", &time_zones_df);
    // --8<-- [end:example]

    // --8<-- [start:example2]
    let time_zones_operations = time_zones_df
        .lazy()
        .select([
            col("tz_aware")
                .dt()
                .replace_time_zone(
                    Some("Europe/Brussels".to_string()),
                    lit("raise"),
                    NonExistent::Raise,
                )
                .alias("replace time zone"),
            col("tz_aware")
                .dt()
                .convert_time_zone("Asia/Kathmandu".to_string())
                .alias("convert time zone"),
            col("tz_aware")
                .dt()
                .replace_time_zone(None, lit("raise"), NonExistent::Raise)
                .alias("unset time zone"),
        ])
        .collect()?;
    println!("{}", &time_zones_operations);
    // --8<-- [end:example2]

    Ok(())
}
