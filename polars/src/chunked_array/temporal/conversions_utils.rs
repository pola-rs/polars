// Conversion extracted from:
// https://docs.rs/arrow/1.0.0/src/arrow/array/array.rs.html#589

use chrono::{NaiveDateTime, NaiveTime, Timelike};

/// Number of seconds in a day
const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
const MILLISECONDS_IN_SECOND: i64 = 1_000;
/// Number of microseconds in a second
const MICROSECONDS_IN_SECOND: i64 = 1_000_000;
/// Number of nanoseconds in a second
const NANOSECONDS_IN_SECOND: i64 = 1_000_000_000;

pub(crate) fn date32_as_datetime(v: i32) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(v as i64 * SECONDS_IN_DAY, 0)
}

pub(crate) fn date64_as_datetime(v: i64) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(
        // extract seconds from milliseconds
        v / MILLISECONDS_IN_SECOND,
        // discard extracted seconds and convert milliseconds to nanoseconds
        (v % MILLISECONDS_IN_SECOND * MICROSECONDS_IN_SECOND) as u32,
    )
}

pub(crate) fn timestamp_nanoseconds_as_datetime(v: i64) -> NaiveDateTime {
    // some nanoseconds will be truncated down as integer division rounds downwards
    let seconds = v / 1_000_000_000;
    // we can use that to compute the remaining nanoseconds
    let nanoseconds = (v - (seconds * 1_000_000_000)) as u32;

    NaiveDateTime::from_timestamp(seconds, nanoseconds)
}

pub(crate) fn timestamp_microseconds_as_datetime(v: i64) -> NaiveDateTime {
    // see nanoseconds for the logic
    let seconds = v / 1_000_000;
    let microseconds = (v - (seconds * 1_000_000)) as u32;

    NaiveDateTime::from_timestamp(seconds, microseconds)
}

pub(crate) fn timestamp_milliseconds_as_datetime(v: i64) -> NaiveDateTime {
    // see nanoseconds for the logic
    let seconds = v / 1000;
    let milliseconds = (v - (seconds * 1000)) as u32;

    NaiveDateTime::from_timestamp(seconds, milliseconds)
}

pub(crate) fn timestamp_seconds_as_datetime(seconds: i64) -> NaiveDateTime {
    NaiveDateTime::from_timestamp(seconds, 0)
}

// date64 is number of milliseconds since the Unix Epoch
pub fn naive_datetime_to_date64(v: &NaiveDateTime) -> i64 {
    v.timestamp_millis()
}

pub fn naive_datetime_to_date32(v: &NaiveDateTime) -> i32 {
    (naive_datetime_to_date64(v) / (MILLISECONDS_IN_SECOND * SECONDS_IN_DAY)) as i32
}

pub(crate) fn naive_datetime_to_timestamp_nanoseconds(v: &NaiveDateTime) -> i64 {
    v.timestamp_nanos()
}

pub(crate) fn naive_datetime_to_timestamp_microseconds(v: &NaiveDateTime) -> i64 {
    v.timestamp() * 1_000_000 + v.timestamp_subsec_micros() as i64
}

pub(crate) fn naive_datetime_to_timestamp_milliseconds(v: &NaiveDateTime) -> i64 {
    v.timestamp_millis()
}

pub(crate) fn naive_datetime_to_timestamp_seconds(v: &NaiveDateTime) -> i64 {
    v.timestamp()
}

pub(crate) fn naive_time_to_time64_nanoseconds(v: &NaiveTime) -> i64 {
    // 3600 seconds in an hour
    v.hour() as i64 * 3600 * NANOSECONDS_IN_SECOND
        // 60 seconds in a minute
        + v.minute() as i64 * 60 * NANOSECONDS_IN_SECOND
        + v.second() as i64 * NANOSECONDS_IN_SECOND
        + v.nanosecond() as i64
}

pub(crate) fn naive_time_to_time64_microseconds(v: &NaiveTime) -> i64 {
    v.hour() as i64 * 3600 * MICROSECONDS_IN_SECOND
        + v.minute() as i64 * 60 * MICROSECONDS_IN_SECOND
        + v.second() as i64 * MICROSECONDS_IN_SECOND
        + v.nanosecond() as i64 / 1000
}

pub(crate) fn naive_time_to_time32_milliseconds(v: &NaiveTime) -> i32 {
    v.hour() as i32 * 3600 * MILLISECONDS_IN_SECOND as i32
        + v.minute() as i32 * 60 * MILLISECONDS_IN_SECOND as i32
        + v.second() as i32 * MILLISECONDS_IN_SECOND as i32
        + v.nanosecond() as i32 / 1000_000
}

pub(crate) fn naive_time_to_time32_seconds(v: &NaiveTime) -> i32 {
    v.hour() as i32 * 3600 + v.minute() as i32 * 60 + v.second() as i32 + v.nanosecond() as i32
}
pub(crate) fn time64_nanosecond_as_time(v: i64) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight(
        // extract seconds from nanoseconds
        (v / NANOSECONDS_IN_SECOND) as u32,
        // discard extracted seconds
        (v % NANOSECONDS_IN_SECOND) as u32,
    )
}

pub(crate) fn time64_microsecond_as_time(v: i64) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight(
        // extract seconds from microseconds
        (v / MICROSECONDS_IN_SECOND) as u32,
        // discard extracted seconds and convert microseconds to
        // nanoseconds
        (v % MICROSECONDS_IN_SECOND * MILLISECONDS_IN_SECOND) as u32,
    )
}

pub(crate) fn time32_second_as_time(v: i32) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight(v as u32, 0)
}

pub(crate) fn time32_millisecond_as_time(v: i32) -> NaiveTime {
    let v = v as u32;
    NaiveTime::from_num_seconds_from_midnight(
        // extract seconds from milliseconds
        v / MILLISECONDS_IN_SECOND as u32,
        // discard extracted seconds and convert milliseconds to
        // nanoseconds
        v % MILLISECONDS_IN_SECOND as u32 * MICROSECONDS_IN_SECOND as u32,
    )
}
