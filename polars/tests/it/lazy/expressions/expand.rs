use polars::export::chrono::NaiveDate;

use super::*;

#[test]
fn test_expand_datetimes_3042() -> PolarsResult<()> {
    let low = NaiveDate::from_ymd(2020, 1, 1).and_hms(0, 0, 0);
    let high = NaiveDate::from_ymd(2020, 2, 1).and_hms(0, 0, 0);
    let date_range = polars_time::date_range(
        "dt1",
        low,
        high,
        Duration::parse("1w"),
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
        None,
    )
    .into_series();

    let out = df![
        "dt1" => date_range.clone(),
        "dt2" => date_range,
    ]?
    .lazy()
    .with_column(
        dtype_col(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .dt()
            .strftime("%m/%d/%Y"),
    )
    .limit(3)
    .collect()?;

    let expected = df![
        "dt1" => ["01/01/2020", "01/08/2020", "01/15/2020"],
        "dt2" => ["01/01/2020", "01/08/2020", "01/15/2020"],
    ]?;
    assert!(out.frame_equal(&expected));

    Ok(())
}
