use chrono::{Datelike, NaiveDateTime, NaiveTime};
use polars_core::chunked_array::temporal::time_to_time64ns;
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::prelude::*;
#[cfg(feature = "timezones")]
use crate::utils::localize_timestamp;

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

#[doc(hidden)]
pub fn date_range_impl(
    name: &str,
    start: i64,
    stop: i64,
    every: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    _tz: Option<&TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    if start > stop {
        polars_bail!(ComputeError: "'start' cannot be greater than 'stop'")
    }
    if every.negative {
        polars_bail!(ComputeError: "'interval' cannot be negative")
    }
    let mut out = match _tz {
        #[cfg(feature = "timezones")]
        Some(tz) => match tz.parse::<chrono_tz::Tz>() {
            Ok(tz) => {
                let start = localize_timestamp(start, tu, tz);
                let stop = localize_timestamp(stop, tu, tz);
                Int64Chunked::new_vec(
                    name,
                    temporal_range_vec(start?, stop?, every, closed, tu, Some(&tz))?,
                )
                .into_datetime(tu, _tz.cloned())
            }
            Err(_) => polars_bail!(ComputeError: "unable to parse time zone: '{}'", tz),
        },
        _ => Int64Chunked::new_vec(
            name,
            temporal_range_vec(start, stop, every, closed, tu, None)?,
        )
        .into_datetime(tu, None),
    };

    out.set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// Create a [`DatetimeChunked`] from a given `start` and `stop` date and a given `every` interval.
pub fn date_range(
    name: &str,
    start: NaiveDateTime,
    stop: NaiveDateTime,
    every: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    let (start, stop) = match tu {
        TimeUnit::Nanoseconds => (start.timestamp_nanos(), stop.timestamp_nanos()),
        TimeUnit::Microseconds => (start.timestamp_micros(), stop.timestamp_micros()),
        TimeUnit::Milliseconds => (start.timestamp_millis(), stop.timestamp_millis()),
    };
    date_range_impl(name, start, stop, every, closed, tu, tz.as_ref())
}

#[doc(hidden)]
pub fn time_range_impl(
    name: &str,
    start: i64,
    stop: i64,
    every: Duration,
    closed: ClosedWindow,
) -> PolarsResult<TimeChunked> {
    if start > stop {
        polars_bail!(ComputeError: "'start' cannot be greater than 'stop'")
    }
    if every.negative {
        polars_bail!(ComputeError: "'interval' cannot be negative")
    }
    let mut out = Int64Chunked::new_vec(
        name,
        temporal_range_vec(start, stop, every, closed, TimeUnit::Nanoseconds, None)?,
    )
    .into_time();

    out.set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// Create a [`TimeChunked`] from a given `start` and `stop` date and a given `every` interval.
pub fn time_range(
    name: &str,
    start: NaiveTime,
    stop: NaiveTime,
    every: Duration,
    closed: ClosedWindow,
) -> PolarsResult<TimeChunked> {
    let start = time_to_time64ns(&start);
    let stop = time_to_time64ns(&stop);
    time_range_impl(name, start, stop, every, closed)
}

#[cfg(test)]
mod test {
    use chrono::NaiveDate;

    use super::*;
    #[test]
    fn test_date_range_9413() {
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
    }
}
