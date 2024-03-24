// --8<-- [start:setup]
use chrono::prelude::*;
use polars::io::prelude::*;
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let df = CsvReader::from_path("docs/data/apple_stock.csv")
        .unwrap()
        .with_try_parse_dates(true)
        .finish()
        .unwrap();
    println!("{}", &df);
    // --8<-- [end:df]

    // --8<-- [start:filter]
    let filtered_df = df
        .clone()
        .lazy()
        .filter(col("Date").eq(lit(NaiveDate::from_ymd_opt(1995, 10, 16).unwrap())))
        .collect()?;
    println!("{}", &filtered_df);
    // --8<-- [end:filter]

    // --8<-- [start:range]
    let filtered_range_df = df
        .clone()
        .lazy()
        .filter(
            col("Date")
                .gt(lit(NaiveDate::from_ymd_opt(1995, 7, 1).unwrap()))
                .and(col("Date").lt(lit(NaiveDate::from_ymd_opt(1995, 11, 1).unwrap()))),
        )
        .collect()?;
    println!("{}", &filtered_range_df);
    // --8<-- [end:range]

    // --8<-- [start:negative]
    let negative_dates_df = df!(
	"ts"=> &["-1300-05-23", "-1400-03-02"],
	"values"=> &[3, 4])?
    .lazy()
    .with_column(col("ts").str().to_date(StrptimeOptions::default()))
    .collect()?;

    let negative_dates_filtered_df = negative_dates_df
        .clone()
        .lazy()
        .filter(col("ts").dt().year().lt(-1300))
        .collect()?;
    println!("{}", &negative_dates_filtered_df);
    // --8<-- [end:negative]

    Ok(())
}
