//! Utilities for converting dates, times, datetimes, and so on.

use polars::datatypes::TimeUnit;
use polars_core::export::chrono::{NaiveDateTime, NaiveTime, TimeDelta};

pub fn elapsed_offset_to_timedelta(elapsed: i64, time_unit: TimeUnit) -> TimeDelta {
    let (in_second, nano_multiplier) = match time_unit {
        TimeUnit::Nanoseconds => (1_000_000_000, 1),
        TimeUnit::Microseconds => (1_000_000, 1_000),
        TimeUnit::Milliseconds => (1_000, 1_000_000),
    };
    let mut elapsed_sec = elapsed / in_second;
    let mut elapsed_nanos = nano_multiplier * (elapsed % in_second);
    if elapsed_nanos < 0 {
        // TimeDelta expects nanos to always be positive.
        elapsed_sec -= 1;
        elapsed_nanos += 1_000_000_000;
    }
    TimeDelta::new(elapsed_sec, elapsed_nanos as u32).unwrap()
}

/// Convert time-units-since-epoch to a more structured object.
pub fn timestamp_to_naive_datetime(since_epoch: i64, time_unit: TimeUnit) -> NaiveDateTime {
    NaiveDateTime::UNIX_EPOCH + elapsed_offset_to_timedelta(since_epoch, time_unit)
}

/// Convert nanoseconds-since-midnight to a more structured object.
pub fn nanos_since_midnight_to_naivetime(nanos_since_midnight: i64) -> NaiveTime {
    NaiveTime::from_hms_opt(0, 0, 0).unwrap()
        + elapsed_offset_to_timedelta(nanos_since_midnight, TimeUnit::Nanoseconds)
}
