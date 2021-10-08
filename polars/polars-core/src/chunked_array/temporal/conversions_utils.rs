// Conversion extracted from:
// https://docs.rs/arrow/1.0.0/src/arrow/array/array.rs.html#589

use chrono::NaiveDateTime;

/// Number of seconds in a day
pub(crate) const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
const MILLISECONDS_IN_SECOND: i64 = 1_000;
/// Number of microseconds in a second
const MICROSECONDS_IN_SECOND: i64 = 1_000_000;

pub(crate) fn date_as_datetime(v: i32) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(v as i64 * SECONDS_IN_DAY, 0)
}

pub(crate) fn datetime_as_datetime(v: i64) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(
        // extract seconds from milliseconds
        v / MILLISECONDS_IN_SECOND,
        // discard extracted seconds and convert milliseconds to nanoseconds
        (v % MILLISECONDS_IN_SECOND * MICROSECONDS_IN_SECOND) as u32,
    )
}

// datetimeis number of milliseconds since the Unix Epoch
pub fn naive_datetime_to_datetime(v: &NaiveDateTime) -> i64 {
    v.timestamp_millis()
}

pub fn naive_datetime_to_date(v: &NaiveDateTime) -> i32 {
    (naive_datetime_to_datetime(v) / (MILLISECONDS_IN_SECOND * SECONDS_IN_DAY)) as i32
}
