use chrono::{NaiveDateTime, NaiveTime};

// Conversion extracted from:
// https://docs.rs/arrow/1.0.0/src/arrow/array/array.rs.html#589

/// Number of seconds in a day
const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
const MILLISECONDS_IN_SECOND: i64 = 1_000;
/// Number of microseconds in a second
const MICROSECONDS_IN_SECOND: i64 = 1_000_000;
/// Number of nanoseconds in a second
const NANOSECONDS_IN_SECOND: i64 = 1_000_000_000;

pub fn date32_as_datetime(v: i32) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(v as i64 * SECONDS_IN_DAY, 0)
}

pub fn date64_as_datetime(v: i64) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(
        // extract seconds from milliseconds
        v / MILLISECONDS_IN_SECOND,
        // discard extracted seconds and convert milliseconds to nanoseconds
        (v % MILLISECONDS_IN_SECOND * MICROSECONDS_IN_SECOND) as u32,
    )
}

pub fn time64_ns_as_time(v: i64) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight(
        // extract seconds from nanoseconds
        (v / NANOSECONDS_IN_SECOND) as u32,
        // discard extracted seconds
        (v % NANOSECONDS_IN_SECOND) as u32,
    )
}
