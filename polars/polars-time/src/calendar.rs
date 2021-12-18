use crate::groupby::ClosedWindow;
use crate::unit::TimeNanoseconds;
use crate::Duration;
use chrono::NaiveDateTime;

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

pub fn timestamp_ns_to_datetime(v: i64) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(
        // extract seconds from nanoseconds
        v / NS_SECOND,
        // discard extracted seconds
        (v % NS_SECOND) as u32,
    )
}

pub fn date_range(
    start: TimeNanoseconds,
    stop: TimeNanoseconds,
    every: Duration,
    closed: ClosedWindow,
) -> Vec<TimeNanoseconds> {
    let size = ((stop - start) / every.duration() + 1) as usize;
    let mut ts = Vec::with_capacity(size);

    let mut t = start;
    match closed {
        ClosedWindow::Both => {
            while t <= stop {
                ts.push(t);
                t += every.duration()
            }
        }
        ClosedWindow::Left => {
            while t < stop {
                ts.push(t);
                t += every.duration()
            }
        }
        ClosedWindow::Right => {
            t += every.duration();
            while t <= stop {
                ts.push(t);
                t += every.duration()
            }
        }
        ClosedWindow::None => {
            t += every.duration();
            while t < stop {
                ts.push(t);
                t += every.duration()
            }
        }
    }
    debug_assert!(size >= ts.len());
    ts
}
