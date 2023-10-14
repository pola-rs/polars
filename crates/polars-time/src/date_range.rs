use arrow::legacy::time_zone::Tz;
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

/// Create a [`DatetimeChunked`] from a given `start` and `end` date and a given `interval`.
pub fn date_range(
    name: &str,
    start: NaiveDateTime,
    end: NaiveDateTime,
    interval: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    let (start, end) = match tu {
        TimeUnit::Nanoseconds => (
            start.timestamp_nanos_opt().unwrap(),
            end.timestamp_nanos_opt().unwrap(),
        ),
        TimeUnit::Microseconds => (start.timestamp_micros(), end.timestamp_micros()),
        TimeUnit::Milliseconds => (start.timestamp_millis(), end.timestamp_millis()),
    };
    datetime_range_impl(name, start, end, interval, closed, tu, tz.as_ref())
}

#[doc(hidden)]
pub fn datetime_range_impl(
    name: &str,
    start: i64,
    end: i64,
    interval: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    _tz: Option<&TimeZone>,
) -> PolarsResult<DatetimeChunked> {
    let mut out = match _tz {
        #[cfg(feature = "timezones")]
        Some(tz) => match tz.parse::<chrono_tz::Tz>() {
            Ok(tz) => {
                let start = localize_timestamp(start, tu, tz);
                let end = localize_timestamp(end, tu, tz);
                Int64Chunked::new_vec(
                    name,
                    datetime_range_i64(start?, end?, interval, closed, tu, Some(&tz))?,
                )
                .into_datetime(tu, _tz.cloned())
            },
            Err(_) => polars_bail!(ComputeError: "unable to parse time zone: '{}'", tz),
        },
        _ => Int64Chunked::new_vec(
            name,
            datetime_range_i64(start, end, interval, closed, tu, None)?,
        )
        .into_datetime(tu, None),
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
    check_range_bounds(start, end, interval)?;

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

    let mut t = start;
    match closed {
        ClosedWindow::Both => {
            while t <= end {
                ts.push(t);
                t = offset_fn(&interval, t, tz)?
            }
        },
        ClosedWindow::Left => {
            while t < end {
                ts.push(t);
                t = offset_fn(&interval, t, tz)?
            }
        },
        ClosedWindow::Right => {
            t = offset_fn(&interval, t, tz)?;
            while t <= end {
                ts.push(t);
                t = offset_fn(&interval, t, tz)?
            }
        },
        ClosedWindow::None => {
            t = offset_fn(&interval, t, tz)?;
            while t < end {
                ts.push(t);
                t = offset_fn(&interval, t, tz)?
            }
        },
    }
    debug_assert!(size >= ts.len());
    Ok(ts)
}

fn check_range_bounds(start: i64, end: i64, interval: Duration) -> PolarsResult<()> {
    polars_ensure!(end >= start, ComputeError: "`end` must be equal to or greater than `start`");
    polars_ensure!(!interval.negative && !interval.is_zero(), ComputeError: "`interval` must be positive");
    Ok(())
}
