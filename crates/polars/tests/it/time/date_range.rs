use polars::export::chrono::NaiveDate;
use polars::prelude::*;
#[allow(unused_imports)]
use polars::time::date_range;

#[test]
fn test_time_units_9413() {
    let start = NaiveDate::from_ymd_opt(2022, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let stop = NaiveDate::from_ymd_opt(2022, 1, 5)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let actual = date_range(
        "date",
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Both,
        TimeUnit::Milliseconds,
        None,
    )
    .map(|date_range| date_range.into_series());
    let result = format!("{:?}", actual);
    let expected = r#"Ok(shape: (5,)
Series: 'date' [datetime[ms]]
[
	2022-01-01 00:00:00
	2022-01-02 00:00:00
	2022-01-03 00:00:00
	2022-01-04 00:00:00
	2022-01-05 00:00:00
])"#;
    assert_eq!(result, expected);
    let actual = date_range(
        "date",
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Both,
        TimeUnit::Microseconds,
        None,
    )
    .map(|date_range| date_range.into_series());
    let result = format!("{:?}", actual);
    let expected = r#"Ok(shape: (5,)
Series: 'date' [datetime[μs]]
[
	2022-01-01 00:00:00
	2022-01-02 00:00:00
	2022-01-03 00:00:00
	2022-01-04 00:00:00
	2022-01-05 00:00:00
])"#;
    assert_eq!(result, expected);
    let actual = date_range(
        "date",
        start,
        stop,
        Duration::parse("1d"),
        ClosedWindow::Both,
        TimeUnit::Nanoseconds,
        None,
    )
    .map(|date_range| date_range.into_series());
    let result = format!("{:?}", actual);
    let expected = r#"Ok(shape: (5,)
Series: 'date' [datetime[ns]]
[
	2022-01-01 00:00:00
	2022-01-02 00:00:00
	2022-01-03 00:00:00
	2022-01-04 00:00:00
	2022-01-05 00:00:00
])"#;
    assert_eq!(result, expected);
    assert_eq!(result, expected);
}
