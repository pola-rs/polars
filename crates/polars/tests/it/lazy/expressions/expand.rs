use jiff::civil::Date as NaiveDate;

use super::*;

#[test]
fn test_expand_datetimes_3042() -> PolarsResult<()> {
    let low = NaiveDate::new(2020, 1, 1).unwrap().at(0, 0, 0, 0);
    let high = NaiveDate::new(2020, 2, 1).unwrap().at(0, 0, 0, 0);
    let date_range_ = polars_time::date_range(
        "dt1".into(),
        low,
        high,
        Duration::parse("1w"),
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
        None,
    )?
    .into_series();

    let out = df![
        "dt1" => date_range_.clone(),
        "dt2" => date_range_,
    ]?
    .lazy()
    .with_column(
        dtype_col(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .as_selector()
            .as_expr()
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
