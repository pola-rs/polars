use super::*;
use polars::export::chrono::prelude::*;

#[test]
#[cfg(all(
    feature = "temporal",
    feature = "dtype-date",
    feature = "dynamic_groupby"
))]
fn test_groupby_dynamic_week_bounds() -> Result<()> {
    let start = NaiveDate::from_ymd(2022, 2, 1).and_hms(0, 0, 0);
    let stop = NaiveDate::from_ymd(2022, 2, 14).and_hms(0, 0, 0);
    let range = date_range(
        "dt",
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
    )
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
                every: Duration::parse("1w"),
                period: Duration::parse("1w"),
                offset: Duration::parse("0w"),
                closed_window: ClosedWindow::Left,
                truncate: false,
                include_boundaries: true,
            },
        )
        .agg([col("a").sum()])
        .collect()?;
    let a = out.column("a")?;
    assert_eq!(a.get(0), AnyValue::Int32(7));
    assert_eq!(a.get(1), AnyValue::Int32(6));
    Ok(())
}
