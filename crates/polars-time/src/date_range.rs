use arrow::legacy::time_zone::Tz;
use chrono::{Datelike, NaiveDateTime, NaiveTime};
use polars_core::chunked_array::temporal::time_to_time64ns;
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::prelude::*;

/// Create a [`DatetimeChunked`] from a given `start`, `end`, `interval`, and `num_samples`.
#[allow(clippy::too_many_arguments)]
pub fn date_range(
    name: PlSmallStr,
    start: Option<NaiveDateTime>,
    end: Option<NaiveDateTime>,
    interval: Option<Duration>,
    num_samples: Option<i64>,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&Tz>,
) -> PolarsResult<DatetimeChunked> {
    macro_rules! extract {
        ($t:ident, $tu:ident) => {
            match $tu {
                TimeUnit::Nanoseconds => $t.and_utc().timestamp_nanos_opt().unwrap(),
                TimeUnit::Microseconds => $t.and_utc().timestamp_micros(),
                TimeUnit::Milliseconds => $t.and_utc().timestamp_millis(),
            }
        };
    }

    match (start, end, interval, num_samples) {
        (Some(start), Some(end), Some(interval), None) => {
            let start = extract!(start, tu);
            let end = extract!(end, tu);
            datetime_range_impl_start_end_interval(name, start, end, interval, closed, tu, tz)
        },
        (Some(start), Some(end), None, Some(num_samples)) => {
            let start = extract!(start, tu);
            let end = extract!(end, tu);
            datetime_range_impl_start_end_samples(name, start, end, num_samples, closed, tu, tz)
        },
        (Some(start), None, Some(interval), Some(num_samples)) => {
            let start = extract!(start, tu);
            datetime_range_impl_start_interval_samples(
                name,
                start,
                interval,
                num_samples,
                closed,
                tu,
                tz,
            )
        },
        (None, Some(end), Some(interval), Some(num_samples)) => {
            let end = extract!(end, tu);
            let out = datetime_range_impl_start_interval_samples(
                name,
                end,
                -interval,
                num_samples,
                closed,
                tu,
                tz,
            )?;
            let out = out.into_physical().reverse();
            match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => Ok(out.into_datetime(tu, Some(TimeZone::from_chrono(tz)))),
                _ => Ok(out.into_datetime(tu, None)),
            }
        },
        _ => {
            polars_bail!(InvalidOperation: "Exactly three of 'start', 'end', 'interval', and 'num_samples' must be supplied.");
        },
    }
}

pub fn in_nanoseconds_window(ndt: &NaiveDateTime) -> bool {
    // ~584 year around 1970
    !(ndt.year() > 2554 || ndt.year() < 1386)
}

#[doc(hidden)]
pub fn datetime_range_impl_start_end_interval(
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
        datetime_range_i64_start_end_interval(start, end, interval, closed, tu, tz)?,
    );
    let mut out = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => out.into_datetime(tu, Some(TimeZone::from_chrono(tz))),
        _ => out.into_datetime(tu, None),
    };

    out.physical_mut().set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

#[doc(hidden)]
pub fn datetime_range_impl_start_interval_samples(
    name: PlSmallStr,
    start: i64,
    interval: Duration,
    num_samples: i64,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&Tz>,
) -> PolarsResult<DatetimeChunked> {
    let out = Int64Chunked::new_vec(
        name,
        datetime_range_i64_start_interval_samples(start, interval, num_samples, closed, tu, tz)?,
    );
    let mut out = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => out.into_datetime(tu, Some(TimeZone::from_chrono(tz))),
        _ => out.into_datetime(tu, None),
    };

    out.physical_mut().set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

#[doc(hidden)]
pub fn datetime_range_impl_start_end_samples(
    name: PlSmallStr,
    start: i64,
    end: i64,
    num_samples: i64,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&Tz>,
) -> PolarsResult<DatetimeChunked> {
    let ascending = start >= end;
    let values = if num_samples == 0 {
        Vec::<i64>::new()
    } else {
        // The bin width depends on the interval closure.
        let divisor = match closed {
            ClosedWindow::None => num_samples + 1,
            ClosedWindow::Left => num_samples,
            ClosedWindow::Right => num_samples,
            ClosedWindow::Both => num_samples - 1,
        };
        let bin_width = (end - start) as f64 / (divisor as f64);

        // For left-open intervals, increase the left by one interval.
        let start = if closed == ClosedWindow::None || closed == ClosedWindow::Right {
            start as f64 + bin_width
        } else {
            start as f64
        };

        let mut values: Vec<i64> = (0..num_samples)
            .map(|x| (x as f64 * bin_width + start) as i64)
            .collect();

        // For right-closed and fully-closed interval, ensure the last point is exact.
        if closed == ClosedWindow::Right || closed == ClosedWindow::Both {
            let last = values.len() - 1;
            values[last] = end;
        }
        values
    };
    let out = Int64Chunked::new_vec(name, values);
    let mut out = match tz {
        #[cfg(feature = "timezones")]
        Some(tz) => out.into_datetime(tu, Some(TimeZone::from_chrono(tz))),
        _ => out.into_datetime(tu, None),
    };

    let sorted = if ascending {
        IsSorted::Ascending
    } else {
        IsSorted::Descending
    };
    out.physical_mut().set_sorted_flag(sorted);
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
        datetime_range_i64_start_end_interval(
            start,
            end,
            interval,
            closed,
            TimeUnit::Nanoseconds,
            None,
        )?,
    )
    .into_time();

    out.physical_mut().set_sorted_flag(IsSorted::Ascending);
    Ok(out)
}

/// vector of i64 representing temporal values
pub(crate) fn datetime_range_i64_start_end_interval(
    mut start: i64,
    mut end: i64,
    interval: Duration,
    closed: ClosedWindow,
    time_unit: TimeUnit,
    time_zone: Option<&Tz>,
) -> PolarsResult<Vec<i64>> {
    let mut step = match time_unit {
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
        if interval.negative {
            step = -step;
        }
        polars_ensure!(
            step != 0,
            InvalidOperation: "interval {} is too small for time unit {} and got rounded down to zero",
            interval,
            time_unit,
        );
        // If step points in the wrong direction, we have no values.
        if (start <= end) != (step > 0) {
            return Ok(Vec::<i64>::new());
        }

        // Update end points based on interval closure.
        if closed == ClosedWindow::Right || closed == ClosedWindow::None {
            start += step;
        };
        if closed == ClosedWindow::Left || closed == ClosedWindow::None {
            end -= step;
        }

        let out = if step < 0 {
            // Negative interval, we move backwards.
            (end..=start)
                .rev()
                .step_by(-step as usize)
                .collect::<Vec<i64>>()
        } else {
            // Positive interval, we move forwards.
            (start..=end).step_by(step as usize).collect::<Vec<i64>>()
        };
        return Ok(out);
    }

    let size = ((end - start) / step + 1) as usize;
    let offset_fn = match time_unit {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    let mut ts = Vec::with_capacity(size);

    // Open the right interval if we're left-closed or right-closed
    if closed == ClosedWindow::Left || closed == ClosedWindow::None {
        if interval.negative {
            end += step;
        } else {
            end -= step;
        }
    }

    if closed == ClosedWindow::Left || closed == ClosedWindow::Both {
        ts.push(start);
    };
    let mut i = 1;
    let mut t = offset_fn(&interval, start, time_zone)?;
    while t <= end {
        ts.push(t);
        i += 1;
        t = offset_fn(&(interval * i), start, time_zone)?;
    }
    debug_assert!(size >= ts.len());
    Ok(ts)
}

pub(crate) fn datetime_range_i64_start_interval_samples(
    mut start: i64,
    interval: Duration,
    num_samples: i64,
    closed: ClosedWindow,
    time_unit: TimeUnit,
    time_zone: Option<&Tz>,
) -> PolarsResult<Vec<i64>> {
    let time_zone_opt: Option<TimeZone> = match time_zone {
        #[cfg(feature = "timezones")]
        Some(tz) => Some(TimeZone::from_chrono(tz)),
        _ => None,
    };
    if interval.is_constant_duration(time_zone_opt.as_ref()) {
        // Fast path
        let mut step = match time_unit {
            TimeUnit::Nanoseconds => interval.duration_ns(),
            TimeUnit::Microseconds => interval.duration_us(),
            TimeUnit::Milliseconds => interval.duration_ms(),
        };
        polars_ensure!(
            step != 0,
            InvalidOperation: "interval {} is too small for time unit {} and got rounded down to zero",
            interval,
            time_unit,
        );

        if interval.negative {
            step = -step;
        }

        // If the interval is left-open, start one interval away.
        if closed == ClosedWindow::Right || closed == ClosedWindow::None {
            start += step;
        }

        let out = if step < 0 {
            // Negative interval, we move backwards.
            (start + (step * num_samples) + 1..=start)
                .rev()
                .step_by((-step) as usize)
                .collect::<Vec<i64>>()
        } else {
            // Positive interval, we move forwards.
            (start..start + step * num_samples)
                .step_by(step as usize)
                .collect::<Vec<i64>>()
        };
        return Ok(out);
    }

    let offset_fn = match time_unit {
        TimeUnit::Nanoseconds => Duration::add_ns,
        TimeUnit::Microseconds => Duration::add_us,
        TimeUnit::Milliseconds => Duration::add_ms,
    };

    // Start with one interval offset if we're not left-closed.
    let start = (closed == ClosedWindow::Right || closed == ClosedWindow::None) as i64;
    let ts = (start..start + num_samples)
        .map(|i| offset_fn(&(interval * i), start, time_zone))
        .collect::<PolarsResult<Vec<i64>>>()?;
    debug_assert!(num_samples as usize == ts.len());
    Ok(ts)
}
