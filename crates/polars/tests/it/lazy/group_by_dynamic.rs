// used only if feature="temporal", "dtype-date", "dynamic_group_by"
#[allow(unused_imports)]
use jiff::civil::Date as NaiveDate;

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
    let start = NaiveDate::new(2022, 2, 1).unwrap().at(0, 0, 0, 0);
    let stop = NaiveDate::new(2022, 2, 14).unwrap().at(0, 0, 0, 0);
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
