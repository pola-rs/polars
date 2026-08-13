use arrow::datatypes::TimeUnit;
use arrow::temporal_conversions::date_to_date32_opt;
pub use arrow::temporal_conversions::{
    EPOCH_DAYS_FROM_CE, MICROSECONDS, MICROSECONDS_IN_DAY, MILLISECONDS, MILLISECONDS_IN_DAY,
    NANOSECONDS, NANOSECONDS_IN_DAY, SECONDS_IN_DAY,
};
use jiff::civil::{Date as NaiveDate, DateTime as NaiveDateTime, Time as NaiveTime};
use jiff::tz::TimeZone;

/// Get the time unit as a multiple of a second
pub const fn time_unit_multiple(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => MILLISECONDS,
        TimeUnit::Microsecond => MICROSECONDS,
        TimeUnit::Nanosecond => NANOSECONDS,
    }
}

/// Parses `value` to `Option<i64>` consistent with the Arrow's definition of timestamp without timezone.
/// Returns in scale `tz` of `TimeUnit`.
#[inline]
pub fn utf8_to_naive_timestamp_scalar(value: &str, fmt: &str, tu: &TimeUnit) -> Option<i64> {
    // "%+" mirrors chrono's combined ISO 8601 / RFC 3339 format specifier,
    // which jiff's strtime engine does not implement as a single specifier;
    // fall back to jiff's native ISO 8601 datetime parser for this case.
    let dt = if fmt == "%+" {
        value.parse::<NaiveDateTime>().ok()?
    } else {
        NaiveDateTime::strptime(fmt, value).ok()?
    };
    let ts = TimeZone::UTC.to_timestamp(dt).ok()?;
    Some(match tu {
        TimeUnit::Second => ts.as_second(),
        TimeUnit::Millisecond => ts.as_millisecond(),
        TimeUnit::Microsecond => ts.as_microsecond(),
        TimeUnit::Nanosecond => i64::try_from(ts.as_nanosecond()).ok()?,
    })
}

/// Parses an ISO-8601 date (`YYYY-MM-DD`) into days since the Unix
/// epoch; non-parsable values return `None`.
#[inline]
pub fn utf8_to_naive_date_scalar(value: &str) -> Option<i32> {
    let d = value.parse::<NaiveDate>().ok()?;
    date_to_date32_opt(d)
}

/// Parses an ISO-8601 time (`HH:MM:SS[.fff]`) into elapsed time since
/// midnight in the given `TimeUnit`; non-parsable values return `None`.
#[inline]
pub fn utf8_to_naive_time_scalar(value: &str, tu: TimeUnit) -> Option<i64> {
    value.parse::<NaiveTime>().ok().map(|t| {
        let secs = t.hour() as i64 * 3_600 + t.minute() as i64 * 60 + t.second() as i64;
        let nanos = t.subsec_nanosecond() as i64;
        match tu {
            TimeUnit::Second => secs,
            TimeUnit::Millisecond => secs * MILLISECONDS + nanos / 1_000_000,
            TimeUnit::Microsecond => secs * MICROSECONDS + nanos / 1_000,
            TimeUnit::Nanosecond => secs * NANOSECONDS + nanos,
        }
    })
}
