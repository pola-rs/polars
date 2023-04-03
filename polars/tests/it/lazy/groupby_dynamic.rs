// used only if feature="temporal", "dtype-date", "dynamic_groupby"
#[allow(unused_imports)]
use polars::export::chrono::prelude::*;

// used only if feature="temporal", "dtype-date", "dynamic_groupby"
#[allow(unused_imports)]
use super::*;

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dtype-date",
    feature = "dynamic_groupby"
))]
fn test_groupby_dynamic_week_bounds() -> PolarsResult<()> {
    let start = NaiveDate::from_ymd_opt(2022, 2, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2022, 2, 14)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let range = polars_time::date_range(
        "dt",
        start,
        stop,
        "1d".parse().unwrap(),
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
        None,
    )?
    .into_series();

    let a = Int32Chunked::full("a", 1, range.len());
    let df = df![
        "dt" => range,
        "a" => a
    ]?;

    let out = df
        .lazy()
        .groupby_dynamic(
            [],
            DynamicGroupOptions {
                index_column: "dt".into(),
                every: "1w".parse().unwrap(),
                period: "1w".parse().unwrap(),
                offset: "0w".parse().unwrap(),
                closed_window: ClosedWindow::Left,
                truncate: false,
                include_boundaries: true,
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
