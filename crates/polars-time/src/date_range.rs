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
    name: PlSmallStr,
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
    name: PlSmallStr,
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
        Some(tz) => out.into_datetime(tu, Some(TimeZone::from_chrono(tz))),
        _ => out.into_datetime(tu, None),
    };

    out.set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// Create a [`TimeChunked`] from a given `start` and `end` date and a given `interval`.
pub fn time_range(
    name: PlSmallStr,
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
    name: PlSmallStr,
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
    time_unit: TimeUnit,
    time_zone: Option<&Tz>,
) -> PolarsResult<Vec<i64>> {
    if start > end {
        return Ok(Vec::new());
    }
    polars_ensure!(
        !interval.negative && !interval.is_zero(),
        ComputeError: "`interval` must be positive"
    );

    let duration = match time_unit {
        TimeUnit::Nanoseconds => interval.duration_ns(),
        TimeUnit::Microseconds => interval.duration_us(),
        TimeUnit::Milliseconds => interval.duration_ms(),
    };
    let time_zone_opt: Option<TimeZone> = match time_zone {
        #[cfg(feature = "timezones")]
        Some(tz) => Some(TimeZone::from_chrono(tz)),
        _ => None,
    };

    if interval.is_constant_duration(time_zone_opt.as_ref()) {
        // Fast path!
        let step: usize = duration.try_into().map_err(
            |_err| polars_err!(ComputeError: "Could not convert {:?} to usize", duration),
        )?;
        polars_ensure!(
            step != 0,
            InvalidOperation: "interval {} is too small for time unit {} and got rounded down to zero",
            interval,
            time_unit,
        );
        return match closed {
            ClosedWindow::Both => Ok((start..=end).step_by(step).collect::<Vec<i64>>()),
            ClosedWindow::None => Ok((start + duration..end).step_by(step).collect::<Vec<i64>>()),
            ClosedWindow::Left => Ok((start..end).step_by(step).collect::<Vec<i64>>()),
            ClosedWindow::Right => Ok((start + duration..=end).step_by(step).collect::<Vec<i64>>()),
        };
    }

    let size = ((end - start) / duration + 1) as usize;
    let offset_fn = match time_unit {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };
    let mut ts = Vec::with_capacity(size);
    let mut i = match closed {
        ClosedWindow::Both | ClosedWindow::Left => 0,
        ClosedWindow::Right | ClosedWindow::None => 1,
    };
    let mut t = offset_fn(&(interval * i), start, time_zone)?;
    i += 1;
    match closed {
        ClosedWindow::Both | ClosedWindow::Right => {
            while t <= end {
                ts.push(t);
                t = offset_fn(&(interval * i), start, time_zone)?;
                i += 1;
            }
        },
        ClosedWindow::Left | ClosedWindow::None => {
            while t < end {
                ts.push(t);
                t = offset_fn(&(interval * i), start, time_zone)?;
                i += 1;
            }
        },
    }
    debug_assert!(size >= ts.len());
    Ok(ts)
}
