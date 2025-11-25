// used only if feature="temporal", "dtype-date", "dynamic_group_by"
#[cfg(feature = "timezones")]
use std::str::FromStr;

#[allow(unused_imports)]
use chrono::prelude::*;

// used only if feature="temporal", "dtype-date", "dynamic_group_by"
#[allow(unused_imports)]
use super::*;

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dtype-date",
    feature = "dynamic_group_by"
))]
fn test_group_by_dynamic_week_bounds() -> PolarsResult<()> {
    let start = NaiveDate::from_ymd_opt(2022, 2, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2022, 2, 14)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let range = polars_time::date_range(
        "dt".into(),
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
        None,
    )?
    .into_series();

    let a = Int32Chunked::full("a".into(), 1, range.len());
    let df = df![
        "dt" => range,
        "a" => a
    ]?;

    let out = df
        .lazy()
        .group_by_dynamic(
            col("dt"),
            [],
            DynamicGroupOptions {
                every: Duration::parse("1w"),
                period: Duration::parse("1w"),
                offset: Duration::parse("0w"),
                closed_window: ClosedWindow::Left,
                label: Label::DataPoint,
                include_boundaries: true,
                start_by: StartBy::DataPoint,
                ..Default::default()
            },
        )
        .agg([col("a").sum()])
        .collect()?;
    let a = out.column("a")?;
    assert_eq!(a.get(0)?, AnyValue::Int32(7));
    assert_eq!(a.get(1)?, AnyValue::Int32(6));
    Ok(())
}

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dynamic_group_by",
    feature = "timezones"
))]
fn test_group_by_dynamic_dst_transition() -> PolarsResult<()> {
    use arrow::legacy::time_zone::Tz;

    // This test reproduces issue #25410
    // When group_by_dynamic calculates window boundaries, those boundaries
    // can fall on non-existent datetimes during DST transitions, causing a panic.
    // In this case, with a 17-day period starting from Feb 7, 2024, the window
    // boundaries will include March 10, 2024 2:00 AM, which doesn't exist in
    // America/New_York due to DST (clocks spring forward to 3:00 AM).

    let tz = Tz::from_str("America/New_York").unwrap();

    // Create datetime range from Feb 7, 2024 9:31 to Feb 24, 2024 16:00
    // with 1-minute intervals in America/New_York timezone
    let start = NaiveDate::from_ymd_opt(2024, 2, 7)
        .unwrap()
        .and_hms_opt(9, 31, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 2, 24)
        .unwrap()
        .and_hms_opt(16, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1m"),
        ClosedWindow::Left,
        TimeUnit::Microseconds,
        Some(&tz),
    )?
    .into_series();

    // Create a simple ID column for cross join
    let id = Int32Chunked::from_slice("id".into(), &[1, 2, 3]);

    // Create the base dataframe
    let df = df![
        "timestamp" => timestamp,
    ]?;

    let id_df = df![
        "id" => id,
    ]?;

    // Cross join to expand the dataset
    let df = df.lazy().cross_join(id_df.lazy(), None).collect()?;

    // This should panic with the error:
    // "datetime '2024-03-10 02:00:00' is non-existent in time zone 'America/New_York'"
    // because the 17-day period causes window boundaries to fall on the DST transition
    let result = df
        .lazy()
        .group_by_dynamic(
            col("timestamp"),
            [col("id")],
            DynamicGroupOptions {
                every: Duration::parse("1m"),
                period: Duration::parse("17d"),
                offset: Duration::parse("0d"),
                closed_window: ClosedWindow::Both,
                label: Label::Left,
                include_boundaries: false,
                start_by: StartBy::DataPoint,
                ..Default::default()
            },
        )
        .agg([col("timestamp").last().alias("ts_last")])
        .collect();

    // Currently this will panic, but ideally it should succeed
    // Once fixed, this assertion should pass
    assert!(result.is_ok());

    Ok(())
}

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dynamic_group_by",
    feature = "timezones"
))]
fn test_group_by_dynamic_dst_transition_nanoseconds() -> PolarsResult<()> {
    use arrow::legacy::time_zone::Tz;

    // Test DST handling with nanosecond precision
    let tz = Tz::from_str("America/New_York").unwrap();

    let start = NaiveDate::from_ymd_opt(2024, 2, 7)
        .unwrap()
        .and_hms_opt(9, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 2, 24)
        .unwrap()
        .and_hms_opt(10, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1h"),
        ClosedWindow::Left,
        TimeUnit::Nanoseconds,
        Some(&tz),
    )?
    .into_series();

    let df = df!["timestamp" => timestamp]?;

    let result = df
        .lazy()
        .group_by_dynamic(
            col("timestamp"),
            [],
            DynamicGroupOptions {
                every: Duration::parse("1h"),
                period: Duration::parse("17d"),
                offset: Duration::parse("0d"),
                closed_window: ClosedWindow::Both,
                label: Label::Left,
                include_boundaries: false,
                start_by: StartBy::DataPoint,
                ..Default::default()
            },
        )
        .agg([col("timestamp").count().alias("count")])
        .collect();

    assert!(result.is_ok());
    Ok(())
}

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dynamic_group_by",
    feature = "timezones"
))]
fn test_group_by_dynamic_dst_transition_milliseconds() -> PolarsResult<()> {
    use arrow::legacy::time_zone::Tz;

    // Test DST handling with millisecond precision
    let tz = Tz::from_str("America/New_York").unwrap();

    let start = NaiveDate::from_ymd_opt(2024, 2, 7)
        .unwrap()
        .and_hms_opt(9, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 2, 24)
        .unwrap()
        .and_hms_opt(10, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1h"),
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
        Some(&tz),
    )?
    .into_series();

    let df = df!["timestamp" => timestamp]?;

    let result = df
        .lazy()
        .group_by_dynamic(
            col("timestamp"),
            [],
            DynamicGroupOptions {
                every: Duration::parse("1h"),
                period: Duration::parse("17d"),
                offset: Duration::parse("0d"),
                closed_window: ClosedWindow::Both,
                label: Label::Left,
                include_boundaries: false,
                start_by: StartBy::DataPoint,
                ..Default::default()
            },
        )
        .agg([col("timestamp").count().alias("count")])
        .collect();

    assert!(result.is_ok());
    Ok(())
}
