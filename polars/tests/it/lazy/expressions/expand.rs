use super::*;
use polars::export::chrono::{NaiveDate, NaiveDateTime};

#[test]
fn test_expand_datetimes_3042() -> Result<()> {
    let low = NaiveDate::from_ymd(2020, 1, 1).and_hms(0, 0, 0);
    let high = NaiveDate::from_ymd(2020, 2, 1).and_hms(0, 0, 0);
    let date_range = date_range(
        "dt1",
        low,
        high,
        Duration::parse("1w"),
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
    )
    .into_series();

    let out = df![
        "dt1" => date_range.clone(),
        "dt2" => date_range,
    ]?
    .lazy()
    // this tests if we expand datetimes even though the units differ
    .with_column(
        dtype_col(&DataType::Datetime(TimeUnit::Microseconds, None))
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
