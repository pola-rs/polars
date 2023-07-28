use polars_arrow::time_zone::Tz;
use polars_core::prelude::*;

use crate::prelude::*;

const LAST_DAYS_MONTH: [u32; 12] = [
    31, // January:   31,
    28, // February:  28,
    31, // March:     31,
    30, // April:     30,
    31, // May:       31,
    30, // June:      30,
    31, // July:      31,
    31, // August:    31,
    30, // September: 30,
    31, // October:   31,
    30, // November:  30,
    31, // December:  31,
];

pub(crate) const fn last_day_of_month(month: i32) -> u32 {
    // month is 1 indexed
    LAST_DAYS_MONTH[(month - 1) as usize]
}

pub(crate) const fn is_leap_year(year: i32) -> bool {
    year % 400 == 0 || (year % 4 == 0 && year % 100 != 0)
}
/// nanoseconds per unit
pub const NS_MICROSECOND: i64 = 1_000;
pub const NS_MILLISECOND: i64 = 1_000_000;
pub const NS_SECOND: i64 = 1_000_000_000;
pub const NS_MINUTE: i64 = 60 * NS_SECOND;
pub const NS_HOUR: i64 = 60 * NS_MINUTE;
pub const NS_DAY: i64 = 24 * NS_HOUR;
pub const NS_WEEK: i64 = 7 * NS_DAY;

/// vector of i64 representing temporal values
pub fn temporal_range(
    start: i64,
    stop: i64,
    every: Duration,
    closed: ClosedWindow,
    tu: TimeUnit,
    tz: Option<&Tz>,
) -> PolarsResult<Vec<i64>> {
    let size: usize;
    let offset_fn: fn(&Duration, i64, Option<&Tz>) -> PolarsResult<i64>;

    match tu {
        TimeUnit::Nanoseconds => {
            size = ((stop - start) / every.duration_ns() + 1) as usize;
            offset_fn = Duration::add_ns;
        }
        TimeUnit::Microseconds => {
            size = ((stop - start) / every.duration_us() + 1) as usize;
            offset_fn = Duration::add_us;
        }
        TimeUnit::Milliseconds => {
            size = ((stop - start) / every.duration_ms() + 1) as usize;
            offset_fn = Duration::add_ms;
        }
    }
    let mut ts = Vec::with_capacity(size);

    let mut t = start;
    match closed {
        ClosedWindow::Both => {
            while t <= stop {
                ts.push(t);
                t = offset_fn(&every, t, tz)?
            }
        }
        ClosedWindow::Left => {
            while t < stop {
                ts.push(t);
                t = offset_fn(&every, t, tz)?
            }
        }
        ClosedWindow::Right => {
            t = offset_fn(&every, t, tz)?;
            while t <= stop {
                ts.push(t);
                t = offset_fn(&every, t, tz)?
            }
        }
        ClosedWindow::None => {
            t = offset_fn(&every, t, tz)?;
            while t < stop {
                ts.push(t);
                t = offset_fn(&every, t, tz)?
            }
        }
    }
    debug_assert!(size >= ts.len());
    Ok(ts)
}
