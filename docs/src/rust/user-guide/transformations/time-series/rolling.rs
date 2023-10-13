// --8<-- [start:setup]
use chrono::prelude::*;
use polars::io::prelude::*;
use polars::lazy::dsl::GetOutput;
use polars::prelude::*;
use polars::time::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let df = CsvReader::from_path("docs/data/apple_stock.csv")
        .unwrap()
        .with_try_parse_dates(true)
        .finish()
        .unwrap()
        .sort(["Date"], false, true)?;
    println!("{}", &df);
    // --8<-- [end:df]

    // --8<-- [start:group_by]
    let annual_average_df = df
        .clone()
        .lazy()
        .groupby_dynamic(
            col("Date"),
            [],
            DynamicGroupOptions {
                every: Duration::parse("1y"),
                period: Duration::parse("1y"),
                offset: Duration::parse("0"),
                ..Default::default()
            },
        )
        .agg([col("Close").mean()])
        .collect()?;

    let df_with_year = annual_average_df
        .lazy()
        .with_columns([col("Date").dt().year().alias("year")])
        .collect()?;
    println!("{}", &df_with_year);
    // --8<-- [end:group_by]

    // --8<-- [start:group_by_dyn]
    let df = df!(
	"time" => date_range(
	    "time",
	    NaiveDate::from_ymd_opt(2021, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap(),
	    NaiveDate::from_ymd_opt(2021, 12, 31).unwrap().and_hms_opt(0, 0, 0).unwrap(),
	    Duration::parse("1d"),
	    ClosedWindow::Both,
	    TimeUnit::Milliseconds, None)?.cast(&DataType::Date)?)?;

    let out = df
        .clone()
        .lazy()
        .groupby_dynamic(
            col("time"),
            [],
            DynamicGroupOptions {
                every: Duration::parse("1mo"),
                period: Duration::parse("1mo"),
                offset: Duration::parse("0"),
                closed_window: ClosedWindow::Left,
                ..Default::default()
            },
        )
        .agg([
            col("time")
                .cumcount(true) // python example has false
                .reverse()
                .head(Some(3))
                .alias("day/eom"),
            ((col("time").last() - col("time").first()).map(
                // had to use map as .duration().days() is not available
                |s| {
                    Ok(Some(
                        s.duration()?
                            .into_iter()
                            .map(|d| d.map(|v| v / 1000 / 24 / 60 / 60))
                            .collect::<Int64Chunked>()
                            .into_series(),
                    ))
                },
                GetOutput::from_type(DataType::Int64),
            ) + lit(1))
            .alias("days_in_month"),
        ])
        .explode([col("day/eom")])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:group_by_dyn]

    // --8<-- [start:group_by_roll]
    let df = df!(
    "time" => date_range(
        "time",
        NaiveDate::from_ymd_opt(2021, 12, 16).unwrap().and_hms_opt(0, 0, 0).unwrap(),
        NaiveDate::from_ymd_opt(2021, 12, 16).unwrap().and_hms_opt(3, 0, 0).unwrap(),
        Duration::parse("30m"),
        ClosedWindow::Both,
        TimeUnit::Milliseconds, None)?,
            "groups"=> ["a", "a", "a", "b", "b", "a", "a"],
    )?;
    println!("{}", &df);
    // --8<-- [end:group_by_roll]

    // --8<-- [start:group_by_dyn2]
    let out = df
        .clone()
        .lazy()
        .groupby_dynamic(
            col("time"),
            [col("groups")],
            DynamicGroupOptions {
                every: Duration::parse("1h"),
                period: Duration::parse("1h"),
                offset: Duration::parse("0"),
                include_boundaries: true,
                closed_window: ClosedWindow::Both,
                ..Default::default()
            },
        )
        .agg([count()])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:group_by_dyn2]

    Ok(())
}
