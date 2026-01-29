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

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dynamic_group_by",
    feature = "timezones"
))]
fn test_group_by_dynamic_dst_fall_back_25410() -> PolarsResult<()> {
    // DST fall-back (ambiguous times) - Nov 3, 2024 America/New_York
    use arrow::legacy::time_zone::Tz;

    let tz = Tz::from_str("America/New_York").unwrap();
    let start = NaiveDate::from_ymd_opt(2024, 10, 15)
        .unwrap()
        .and_hms_opt(9, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 11, 1)
        .unwrap()
        .and_hms_opt(10, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1h"),
        ClosedWindow::Left,
        TimeUnit::Microseconds,
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
fn test_group_by_dynamic_dst_month_duration_25410() -> PolarsResult<()> {
    // Month-based durations crossing DST boundaries
    use arrow::legacy::time_zone::Tz;

    let tz = Tz::from_str("America/New_York").unwrap();
    let start = NaiveDate::from_ymd_opt(2024, 1, 15)
        .unwrap()
        .and_hms_opt(2, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 2, 15)
        .unwrap()
        .and_hms_opt(2, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Left,
        TimeUnit::Microseconds,
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
                every: Duration::parse("1d"),
                period: Duration::parse("2mo"),
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
fn test_group_by_dynamic_dst_boundary_on_gap_25410() -> PolarsResult<()> {
    // Window boundary lands exactly on DST gap (2:00 AM March 10, 2024)
    use arrow::legacy::time_zone::Tz;

    let tz = Tz::from_str("America/New_York").unwrap();
    let start = NaiveDate::from_ymd_opt(2024, 3, 5)
        .unwrap()
        .and_hms_opt(2, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 3, 8)
        .unwrap()
        .and_hms_opt(2, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1d"),
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
                every: Duration::parse("1d"),
                period: Duration::parse("5d"),
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
fn test_group_by_dynamic_dst_datapoint_uk_25410() -> PolarsResult<()> {
    // Europe/London: 1:00 AM March 31 2024 is in DST gap
    // start + 17d period lands in the gap
    use arrow::legacy::time_zone::Tz;

    let tz = Tz::from_str("Europe/London").unwrap();
    let start = NaiveDate::from_ymd_opt(2024, 3, 14)
        .unwrap()
        .and_hms_opt(1, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 3, 20)
        .unwrap()
        .and_hms_opt(1, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Left,
        TimeUnit::Microseconds,
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
                every: Duration::parse("1d"),
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
fn test_group_by_dynamic_dst_windowbound_25410() -> PolarsResult<()> {
    // America/New_York: 2:00 AM March 10 2024 is in DST gap
    use arrow::legacy::time_zone::Tz;

    let tz = Tz::from_str("America/New_York").unwrap();
    let start = NaiveDate::from_ymd_opt(2024, 3, 5)
        .unwrap()
        .and_hms_opt(2, 30, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 3, 15)
        .unwrap()
        .and_hms_opt(2, 30, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Left,
        TimeUnit::Microseconds,
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
                every: Duration::parse("1d"),
                period: Duration::parse("5d"),
                offset: Duration::parse("0d"),
                closed_window: ClosedWindow::Both,
                label: Label::Left,
                include_boundaries: false,
                start_by: StartBy::WindowBound,
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
fn test_group_by_dynamic_dst_monday_25410() -> PolarsResult<()> {
    use arrow::legacy::time_zone::Tz;

    let tz = Tz::from_str("America/New_York").unwrap();
    let start = NaiveDate::from_ymd_opt(2024, 3, 4)
        .unwrap()
        .and_hms_opt(2, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 3, 18)
        .unwrap()
        .and_hms_opt(2, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Left,
        TimeUnit::Microseconds,
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
                every: Duration::parse("1w"),
                period: Duration::parse("1w"),
                offset: Duration::parse("0d"),
                closed_window: ClosedWindow::Both,
                label: Label::Left,
                include_boundaries: false,
                start_by: StartBy::Monday,
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
fn test_group_by_dynamic_dst_sunday_offset_25410() -> PolarsResult<()> {
    // offset="2h" lands on 2:00 AM March 10 (DST gap)
    use arrow::legacy::time_zone::Tz;

    let tz = Tz::from_str("America/New_York").unwrap();
    let start = NaiveDate::from_ymd_opt(2024, 3, 3)
        .unwrap()
        .and_hms_opt(2, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 3, 17)
        .unwrap()
        .and_hms_opt(2, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Left,
        TimeUnit::Microseconds,
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
                every: Duration::parse("1w"),
                period: Duration::parse("1w"),
                offset: Duration::parse("2h"),
                closed_window: ClosedWindow::Both,
                label: Label::Left,
                include_boundaries: false,
                start_by: StartBy::Sunday,
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
fn test_group_by_dynamic_dst_offset_gap_25410() -> PolarsResult<()> {
    // offset="2h" lands on 2:00 AM March 10 (DST gap)
    use arrow::legacy::time_zone::Tz;

    let tz = Tz::from_str("America/New_York").unwrap();
    let start = NaiveDate::from_ymd_opt(2024, 3, 10)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2024, 3, 10)
        .unwrap()
        .and_hms_opt(12, 0, 0)
        .unwrap();

    let timestamp = polars_time::date_range(
        "timestamp".into(),
        start,
        stop,
        Duration::parse("1h"),
        ClosedWindow::Left,
        TimeUnit::Microseconds,
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
                period: Duration::parse("3h"),
                offset: Duration::parse("2h"),
                closed_window: ClosedWindow::Both,
                label: Label::Left,
                include_boundaries: false,
                start_by: StartBy::WindowBound,
                ..Default::default()
            },
        )
        .agg([col("timestamp").count().alias("count")])
        .collect();

    assert!(result.is_ok());
    Ok(())
}
