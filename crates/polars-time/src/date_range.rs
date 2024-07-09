use arrow::legacy::time_zone::Tz;
use chrono::{Datelike, NaiveDateTime, NaiveTime};
use polars_core::chunked_array::temporal::time_to_time64ns;
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::prelude::*;

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

/// Create a [`DatetimeChunked`] from a given `start` and `end` date and a given `interval`.
pub fn date_range(
    name: &str,
    start: NaiveDateTime,
    end: NaiveDateTime,
    interval: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&Tz>,
) -> PolarsResult<DatetimeChunked> {
    let (start, end) = match tu {
        TimeUnit::Nanoseconds => (
            start.and_utc().timestamp_nanos_opt().unwrap(),
            end.and_utc().timestamp_nanos_opt().unwrap(),
        ),
        TimeUnit::Microseconds => (
            start.and_utc().timestamp_micros(),
            end.and_utc().timestamp_micros(),
        ),
        TimeUnit::Milliseconds => (
            start.and_utc().timestamp_millis(),
            end.and_utc().timestamp_millis(),
        ),
    };
    datetime_range_impl(name, start, end, interval, closed, tu, tz)
}

#[doc(hidden)]
pub fn datetime_range_impl(
    name: &str,
    start: i64,
    end: i64,
    interval: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&Tz>,
) -> PolarsResult<DatetimeChunked> {
    let out = Int64Chunked::new_vec(
        name,
        datetime_range_i64(start, end, interval, closed, tu, tz)?,
    );
    let mut out = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => out.into_datetime(tu, Some(tz.to_string())),
        _ => out.into_datetime(tu, None),
    };

    out.set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// Create a [`TimeChunked`] from a given `start` and `end` date and a given `interval`.
pub fn time_range(
    name: &str,
    start: NaiveTime,
    end: NaiveTime,
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<TimeChunked> {
    let start = time_to_time64ns(&start);
    let end = time_to_time64ns(&end);
    time_range_impl(name, start, end, interval, closed)
}

#[doc(hidden)]
pub fn time_range_impl(
    name: &str,
    start: i64,
    end: i64,
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<TimeChunked> {
    let mut out = Int64Chunked::new_vec(
        name,
        datetime_range_i64(start, end, interval, closed, TimeUnit::Nanoseconds, None)?,
    )
    .into_time();

    out.set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// vector of i64 representing temporal values
pub(crate) fn datetime_range_i64(
    start: i64,
    end: i64,
    interval: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&Tz>,
) -> PolarsResult<Vec<i64>> {
    if start > end {
        return Ok(Vec::new());
    }
    polars_ensure!(
        !interval.negative && !interval.is_zero(),
        ComputeError: "`interval` must be positive"
    );

    let size: usize;
    let offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>;

    match tu {
        TimeUnit::Nanoseconds => {
            size = ((end - start) / interval.duration_ns() + 1) as usize;
            offset_fn = Duration::add_ns;
        },
        TimeUnit::Microseconds => {
            size = ((end - start) / interval.duration_us() + 1) as usize;
            offset_fn = Duration::add_us;
        },
        TimeUnit::Milliseconds => {
            size = ((end - start) / interval.duration_ms() + 1) as usize;
            offset_fn = Duration::add_ms;
        },
    }
    let mut ts = Vec::with_capacity(size);

    let mut i = match closed {
        ClosedWindow::Both | ClosedWindow::Left => 0,
        ClosedWindow::Right | ClosedWindow::None => 1,
    };
    let mut t = offset_fn(&(interval * i), start, tz)?;
    i += 1;
    match closed {
        ClosedWindow::Both | ClosedWindow::Right => {
            while t <= end {
                ts.push(t);
                t = offset_fn(&(interval * i), start, tz)?;
                i += 1;
            }
        },
        ClosedWindow::Left | ClosedWindow::None => {
            while t < end {
                ts.push(t);
                t = offset_fn(&(interval * i), start, tz)?;
                i += 1;
            }
        },
    }
    debug_assert!(size >= ts.len());
    Ok(ts)
}
