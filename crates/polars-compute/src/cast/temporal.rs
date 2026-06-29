use arrow::datatypes::TimeUnit;
pub use arrow::temporal_conversions::{
    EPOCH_DAYS_FROM_CE, MICROSECONDS, MICROSECONDS_IN_DAY, MILLISECONDS, MILLISECONDS_IN_DAY,
    NANOSECONDS, NANOSECONDS_IN_DAY, SECONDS_IN_DAY,
};
use chrono::format::{Parsed, StrftimeItems};
use chrono::{Datelike, NaiveDate, NaiveTime, Timelike};

/// Get the time unit as a multiple of a second
pub const fn time_unit_multiple(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => MILLISECONDS,
        TimeUnit::Microsecond => MICROSECONDS,
        TimeUnit::Nanosecond => NANOSECONDS,
    }
}

/// Parses `value` to `Option<i64>` consistent with the Arrow's definition of timestamp with timezone.
///
/// `tz` must be built from `timezone` (either via [`parse_offset`] or `chrono-tz`).
/// Returns in scale `tz` of `TimeUnit`.
#[inline]
pub fn utf8_to_timestamp_scalar<T: chrono::TimeZone>(
    value: &str,
    fmt: &str,
    tz: &T,
    tu: &TimeUnit,
) -> Option<i64> {
    let mut parsed = Parsed::new();
    let fmt = StrftimeItems::new(fmt);
    let r = chrono::format::parse(&mut parsed, value, fmt).ok();
    if r.is_some() {
        parsed
            .to_datetime()
            .map(|x| x.naive_utc())
            .map(|x| tz.from_utc_datetime(&x))
            .map(|x| match tu {
                TimeUnit::Second => x.timestamp(),
                TimeUnit::Millisecond => x.timestamp_millis(),
                TimeUnit::Microsecond => x.timestamp_micros(),
                TimeUnit::Nanosecond => x.timestamp_nanos_opt().unwrap(),
            })
            .ok()
    } else {
        None
    }
}

/// Parses `value` to `Option<i64>` consistent with the Arrow's definition of timestamp without timezone.
/// Returns in scale `tz` of `TimeUnit`.
#[inline]
pub fn utf8_to_naive_timestamp_scalar(value: &str, fmt: &str, tu: &TimeUnit) -> Option<i64> {
    let fmt = StrftimeItems::new(fmt);
    let mut parsed = Parsed::new();
    chrono::format::parse(&mut parsed, value, fmt.clone()).ok();
    parsed
        .to_naive_datetime_with_offset(0)
        .map(|x| match tu {
            TimeUnit::Second => x.and_utc().timestamp(),
            TimeUnit::Millisecond => x.and_utc().timestamp_millis(),
            TimeUnit::Microsecond => x.and_utc().timestamp_micros(),
            TimeUnit::Nanosecond => x.and_utc().timestamp_nanos_opt().unwrap(),
        })
        .ok()
}

/// Parses an ISO-8601 date (`YYYY-MM-DD`) into days since the Unix
/// epoch; non-parsable values return `None`.
#[inline]
pub fn utf8_to_naive_date_scalar(value: &str) -> Option<i32> {
    value
        .parse::<NaiveDate>()
        .ok()
        .map(|d| d.num_days_from_ce() - EPOCH_DAYS_FROM_CE)
}

/// Parses an ISO-8601 time (`HH:MM:SS[.fff]`) into elapsed time since
/// midnight in the given `TimeUnit`; non-parsable values return `None`.
#[inline]
pub fn utf8_to_naive_time_scalar(value: &str, tu: TimeUnit) -> Option<i64> {
    value.parse::<NaiveTime>().ok().map(|t| {
        let secs = t.num_seconds_from_midnight() as i64;
        let nanos = t.nanosecond() as i64;
        match tu {
            TimeUnit::Second => secs,
            TimeUnit::Millisecond => secs * MILLISECONDS + nanos / 1_000_000,
            TimeUnit::Microsecond => secs * MICROSECONDS + nanos / 1_000,
            TimeUnit::Nanosecond => secs * NANOSECONDS + nanos,
        }
    })
}
