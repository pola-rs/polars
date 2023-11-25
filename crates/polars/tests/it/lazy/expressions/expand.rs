use polars::export::chrono::NaiveDate;

use super::*;

#[test]
fn test_expand_datetimes_3042() -> PolarsResult<()> {
    let low = NaiveDate::from_ymd_opt(2020, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let high = NaiveDate::from_ymd_opt(2020, 2, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let date_range = polars_time::date_range(
        "dt1",
        low,
        high,
        Duration::parse("1w"),
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
        None,
    )?
    .into_series();

    let out = df![
        "dt1" => date_range.clone(),
        "dt2" => date_range,
    ]?
    .lazy()
    .with_column(
        dtype_col(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .dt()
            .to_string("%m/%d/%Y"),
    )
    .limit(3)
    .collect()?;

    let expected = df![
        "dt1" => ["01/01/2020", "01/08/2020", "01/15/2020"],
        "dt2" => ["01/01/2020", "01/08/2020", "01/15/2020"],
    ]?;
    assert!(out.equals(&expected));

    Ok(())
}
