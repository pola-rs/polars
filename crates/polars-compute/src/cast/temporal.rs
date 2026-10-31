use chrono::format::{Parsed, StrftimeItems, parse, parse_and_remainder};
use chrono::{Datelike, NaiveDate, NaiveTime, Timelike};
use polars_arrow::datatypes::TimeUnit;
pub use polars_arrow::temporal_conversions::{
    EPOCH_DAYS_FROM_CE, MICROSECONDS, MICROSECONDS_IN_DAY, MILLISECONDS, MILLISECONDS_IN_DAY,
    NANOSECONDS, NANOSECONDS_IN_DAY, SECONDS_IN_DAY,
};

/// Get the time unit as a multiple of a second
pub const fn time_unit_multiple(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => MILLISECONDS,
        TimeUnit::Microsecond => MICROSECONDS,
        TimeUnit::Nanosecond => NANOSECONDS,
    }
}

/// Parses an ISO 8601 datetime (`YYYY-MM-DD`, then `T`, `t` or a space, then `HH:MM`
/// with optional seconds, fraction and UTC offset) into a timestamp in `tu`. Values
/// with an offset are converted to UTC. Returns `None` unless the whole value matches.
#[inline]
pub fn utf8_to_timestamp_scalar(value: &str, tu: TimeUnit) -> Option<i64> {
    let mut parsed = Parsed::new();
    let rest = parse_and_remainder(&mut parsed, value, StrftimeItems::new("%Y-%m-%d")).ok()?;
    let rest = rest.strip_prefix(['T', 't', ' '])?;
    let mut rest = parse_and_remainder(&mut parsed, rest, StrftimeItems::new("%H:%M")).ok()?;
    if rest.starts_with(':') {
        rest = parse_and_remainder(&mut parsed, rest, StrftimeItems::new(":%S%.f")).ok()?;
    }
    let dt = if rest.is_empty() {
        parsed.to_naive_datetime_with_offset(0).ok()?
    } else {
        parse(&mut parsed, rest, StrftimeItems::new("%#z")).ok()?;
        parsed.to_datetime().ok()?.naive_utc()
    };
    let dt = dt.and_utc();
    match tu {
        TimeUnit::Second => Some(dt.timestamp()),
        TimeUnit::Millisecond => Some(dt.timestamp_millis()),
        TimeUnit::Microsecond => Some(dt.timestamp_micros()),
        TimeUnit::Nanosecond => dt.timestamp_nanos_opt(),
    }
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
