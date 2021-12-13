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
/// nanoseconds per second
pub const NS_SECONDS: i64 = 1_000_000_000;

/// nanosecs per minute
pub const NS_MINUTE: i64 = 60 * NS_SECONDS;

pub fn timestamp_ns_to_datetime(v: i64) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(
        // extract seconds from nanoseconds
        v / NS_SECONDS,
        // discard extracted seconds
        (v % NS_SECONDS) as u32,
    )
}
