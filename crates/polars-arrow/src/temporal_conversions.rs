//! Conversion methods for dates and times.

use jiff::civil::{Date, DateTime, Time};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp, Zoned};
use polars_error::{PolarsResult, polars_err};

use crate::datatypes::TimeUnit;

/// Number of seconds in a day
pub const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
pub const MILLISECONDS: i64 = 1_000;
/// Number of microseconds in a second
pub const MICROSECONDS: i64 = 1_000_000;
/// Number of nanoseconds in a second
pub const NANOSECONDS: i64 = 1_000_000_000;
/// Number of milliseconds in a day
pub const MILLISECONDS_IN_DAY: i64 = SECONDS_IN_DAY * MILLISECONDS;
/// Number of microseconds in a day
pub const MICROSECONDS_IN_DAY: i64 = SECONDS_IN_DAY * MICROSECONDS;
/// Number of nanoseconds in a day
pub const NANOSECONDS_IN_DAY: i64 = SECONDS_IN_DAY * NANOSECONDS;
/// Number of days between 0001-01-01 and 1970-01-01
pub const EPOCH_DAYS_FROM_CE: i32 = 719_163;

#[inline]
fn unix_epoch_date() -> Date {
    Date::constant(1970, 1, 1)
}

/// converts a `i32` representing a `date32` to [`DateTime`]
#[inline]
pub fn date32_to_datetime(v: i32) -> DateTime {
    date32_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i32` representing a `date32` to [`DateTime`]
#[inline]
pub fn date32_to_datetime_opt(v: i32) -> Option<DateTime> {
    date32_to_date_opt(v).map(|d| d.to_datetime(Time::midnight()))
}

/// converts a `i32` representing a `date32` to [`Date`]
#[inline]
pub fn date32_to_date(days: i32) -> Date {
    date32_to_date_opt(days).expect("out-of-range date")
}

/// converts a `i32` representing a `date32` to [`Date`]
#[inline]
pub fn date32_to_date_opt(days: i32) -> Option<Date> {
    let span = jiff::Span::new().try_days(i64::from(days)).ok()?;
    unix_epoch_date().checked_add(span).ok()
}

/// converts a [`Date`] to an `i32` representing a `date32` (days since the
/// Unix epoch).
///
/// Computed via calendar-day arithmetic on [`Date`] directly rather than
/// through [`Timestamp`], since [`Timestamp`]'s representable range is
/// narrower than the full span of dates a [`Date`] can hold (e.g. it can't
/// represent 9999-12-31 at all).
#[inline]
pub fn date_to_date32_opt(date: Date) -> Option<i32> {
    let span = date.since(unix_epoch_date()).ok()?;
    i32::try_from(span.get_days()).ok()
}

/// Elapsed nanoseconds between the Unix epoch and `dt`, computed via
/// calendar (`DateTime`) arithmetic rather than through `Timestamp`.
///
/// `Timestamp`'s representable range is deliberately narrower than
/// `DateTime`'s (it's kept small enough to accommodate an arbitrary UTC
/// offset applied to it), so routing through `TimeZone::to_timestamp`/
/// `Offset::to_timestamp` would incorrectly reject legitimate `DateTime`
/// values near the -9999/9999 year boundary that `DateTime` itself can
/// represent just fine (e.g. `DateTime::MAX`).
#[inline]
pub fn datetime_to_epoch_nanos_opt(dt: DateTime) -> Option<i128> {
    let diff = jiff::civil::DateTimeDifference::new(unix_epoch_date().to_datetime(Time::midnight()))
        .smallest(jiff::Unit::Nanosecond)
        .largest(jiff::Unit::Second);
    let span = dt.since(diff).ok()?;
    Some(
        i128::from(span.get_seconds()) * 1_000_000_000
            + i128::from(span.get_milliseconds()) * 1_000_000
            + i128::from(span.get_microseconds()) * 1_000
            + i128::from(span.get_nanoseconds()),
    )
}

/// The inverse of [`datetime_to_epoch_nanos_opt`]: converts elapsed
/// nanoseconds since the Unix epoch into a [`DateTime`], via calendar
/// arithmetic rather than through [`Timestamp`] (same rationale as
/// [`datetime_to_epoch_nanos_opt`] - `Timestamp`'s range is too narrow to
/// hold every value `DateTime` can represent). This is on a hot path (it
/// backs every epoch-tick-to-civil conversion), so the time-of-day
/// component is computed with plain arithmetic rather than `Span`/
/// `checked_add`.
#[inline]
pub fn epoch_nanos_to_datetime_opt(epoch_nanos: i128) -> Option<DateTime> {
    const NANOS_PER_DAY: i64 = 86_400_000_000_000;
    let days = epoch_nanos.div_euclid(i128::from(NANOS_PER_DAY));
    let nanos_of_day = epoch_nanos.rem_euclid(i128::from(NANOS_PER_DAY)) as i64;
    let date = date32_to_date_opt(i32::try_from(days).ok()?)?;
    let hour = (nanos_of_day / 3_600_000_000_000) as i8;
    let minute = ((nanos_of_day / 60_000_000_000) % 60) as i8;
    let second = ((nanos_of_day / 1_000_000_000) % 60) as i8;
    let subsec_nanosecond = (nanos_of_day % 1_000_000_000) as i32;
    let time = Time::new(hour, minute, second, subsec_nanosecond).ok()?;
    Some(date.to_datetime(time))
}

/// converts a `i64` representing a `date64` to [`DateTime`]
#[inline]
pub fn date64_to_datetime(v: i64) -> DateTime {
    date64_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

#[inline]
fn date64_to_datetime_opt(v: i64) -> Option<DateTime> {
    Timestamp::from_millisecond(v)
        .ok()
        .map(|ts| TimeZone::UTC.to_datetime(ts))
}

/// converts a `i64` representing a `date64` to [`Date`]
#[inline]
pub fn date64_to_date(milliseconds: i64) -> Date {
    date64_to_datetime(milliseconds).date()
}

/// converts a `i32` representing a `time32(s)` to [`Time`]
#[inline]
pub fn time32s_to_time(v: i32) -> Time {
    Time::midnight()
        .checked_add(jiff::Span::new().seconds(i64::from(v)))
        .expect("invalid time")
}

/// converts a `i64` representing a `duration(s)` to [`SignedDuration`]
#[inline]
pub fn duration_s_to_duration(v: i64) -> SignedDuration {
    SignedDuration::from_secs(v)
}

/// converts a `i64` representing a `duration(ms)` to [`SignedDuration`]
#[inline]
pub fn duration_ms_to_duration(v: i64) -> SignedDuration {
    SignedDuration::from_millis(v)
}

/// converts a `i64` representing a `duration(us)` to [`SignedDuration`]
#[inline]
pub fn duration_us_to_duration(v: i64) -> SignedDuration {
    SignedDuration::from_micros(v)
}

/// converts a `i64` representing a `duration(ns)` to [`SignedDuration`]
#[inline]
pub fn duration_ns_to_duration(v: i64) -> SignedDuration {
    SignedDuration::from_nanos(v)
}

/// converts a `i32` representing a `time32(ms)` to [`Time`]
#[inline]
pub fn time32ms_to_time(v: i32) -> Time {
    let v = v as i64;
    let seconds = v / MILLISECONDS;
    let milli_to_nano = 1_000_000;
    let nano = (v - seconds * MILLISECONDS) * milli_to_nano;
    Time::midnight()
        .checked_add(jiff::Span::new().seconds(seconds).nanoseconds(nano))
        .expect("invalid time")
}

/// converts a `i64` representing a `time64(us)` to [`Time`]
#[inline]
pub fn time64us_to_time(v: i64) -> Time {
    time64us_to_time_opt(v).expect("invalid time")
}

/// converts a `i64` representing a `time64(us)` to [`Time`]
#[inline]
pub fn time64us_to_time_opt(v: i64) -> Option<Time> {
    Time::midnight()
        .checked_add(jiff::Span::new().microseconds(v))
        .ok()
}

/// converts a `i64` representing a `time64(ns)` to [`Time`]
#[inline]
pub fn time64ns_to_time(v: i64) -> Time {
    time64ns_to_time_opt(v).expect("invalid time")
}

/// converts a `i64` representing a `time64(ns)` to [`Time`]
#[inline]
pub fn time64ns_to_time_opt(v: i64) -> Option<Time> {
    Time::midnight()
        .checked_add(jiff::Span::new().nanoseconds(v))
        .ok()
}

/// converts a `i64` representing a `timestamp(s)` to [`DateTime`]
#[inline]
pub fn timestamp_s_to_datetime(seconds: i64) -> DateTime {
    timestamp_s_to_datetime_opt(seconds).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(s)` to [`DateTime`]
#[inline]
pub fn timestamp_s_to_datetime_opt(seconds: i64) -> Option<DateTime> {
    Timestamp::from_second(seconds)
        .ok()
        .map(|ts| TimeZone::UTC.to_datetime(ts))
}

/// converts a `i64` representing a `timestamp(ms)` to [`DateTime`]
#[inline]
pub fn timestamp_ms_to_datetime(v: i64) -> DateTime {
    timestamp_ms_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(ms)` to [`DateTime`]
#[inline]
pub fn timestamp_ms_to_datetime_opt(v: i64) -> Option<DateTime> {
    epoch_nanos_to_datetime_opt(i128::from(v) * 1_000_000)
}

/// converts a `i64` representing a `timestamp(us)` to [`DateTime`]
#[inline]
pub fn timestamp_us_to_datetime(v: i64) -> DateTime {
    timestamp_us_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(us)` to [`DateTime`]
#[inline]
pub fn timestamp_us_to_datetime_opt(v: i64) -> Option<DateTime> {
    epoch_nanos_to_datetime_opt(i128::from(v) * 1_000)
}

/// converts a `i64` representing a `timestamp(ns)` to [`DateTime`]
#[inline]
pub fn timestamp_ns_to_datetime(v: i64) -> DateTime {
    timestamp_ns_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(ns)` to [`DateTime`]
#[inline]
pub fn timestamp_ns_to_datetime_opt(v: i64) -> Option<DateTime> {
    Timestamp::from_nanosecond(i128::from(v))
        .ok()
        .map(|ts| TimeZone::UTC.to_datetime(ts))
}

/// Converts a timestamp in `time_unit` into a [`Timestamp`].
#[inline]
pub(crate) fn timestamp_to_timestamp(timestamp: i64, time_unit: TimeUnit) -> Timestamp {
    let result = match time_unit {
        TimeUnit::Second => Timestamp::from_second(timestamp),
        TimeUnit::Millisecond => Timestamp::from_millisecond(timestamp),
        TimeUnit::Microsecond => Timestamp::from_microsecond(timestamp),
        TimeUnit::Nanosecond => Timestamp::from_nanosecond(i128::from(timestamp)),
    };
    result.expect("invalid or out-of-range timestamp")
}

/// Fallible variant of [`timestamp_to_timestamp`], returning `None` on an
/// out-of-range value instead of panicking.
#[inline]
pub(crate) fn timestamp_to_timestamp_opt(timestamp: i64, time_unit: TimeUnit) -> Option<Timestamp> {
    match time_unit {
        TimeUnit::Second => Timestamp::from_second(timestamp),
        TimeUnit::Millisecond => Timestamp::from_millisecond(timestamp),
        TimeUnit::Microsecond => Timestamp::from_microsecond(timestamp),
        TimeUnit::Nanosecond => Timestamp::from_nanosecond(i128::from(timestamp)),
    }
    .ok()
}

/// Converts a timestamp in `time_unit` into a naive (timezone-less, UTC-labelled) [`DateTime`].
#[inline]
pub(crate) fn timestamp_to_naive_datetime(timestamp: i64, time_unit: TimeUnit) -> DateTime {
    TimeZone::UTC.to_datetime(timestamp_to_timestamp(timestamp, time_unit))
}

/// Converts a timestamp in `time_unit` and `timezone` into a [`Zoned`] datetime.
#[inline]
pub fn timestamp_to_datetime(timestamp: i64, time_unit: TimeUnit, timezone: &TimeZone) -> Zoned {
    timestamp_to_timestamp(timestamp, time_unit).to_zoned(timezone.clone())
}

/// Calculates the scale factor between two TimeUnits. The function returns the
/// scale that should multiply the TimeUnit "b" to have the same time scale as
/// the TimeUnit "a".
pub fn timeunit_scale(a: TimeUnit, b: TimeUnit) -> f64 {
    match (a, b) {
        (TimeUnit::Second, TimeUnit::Second) => 1.0,
        (TimeUnit::Second, TimeUnit::Millisecond) => 0.001,
        (TimeUnit::Second, TimeUnit::Microsecond) => 0.000_001,
        (TimeUnit::Second, TimeUnit::Nanosecond) => 0.000_000_001,
        (TimeUnit::Millisecond, TimeUnit::Second) => 1_000.0,
        (TimeUnit::Millisecond, TimeUnit::Millisecond) => 1.0,
        (TimeUnit::Millisecond, TimeUnit::Microsecond) => 0.001,
        (TimeUnit::Millisecond, TimeUnit::Nanosecond) => 0.000_001,
        (TimeUnit::Microsecond, TimeUnit::Second) => 1_000_000.0,
        (TimeUnit::Microsecond, TimeUnit::Millisecond) => 1_000.0,
        (TimeUnit::Microsecond, TimeUnit::Microsecond) => 1.0,
        (TimeUnit::Microsecond, TimeUnit::Nanosecond) => 0.001,
        (TimeUnit::Nanosecond, TimeUnit::Second) => 1_000_000_000.0,
        (TimeUnit::Nanosecond, TimeUnit::Millisecond) => 1_000_000.0,
        (TimeUnit::Nanosecond, TimeUnit::Microsecond) => 1_000.0,
        (TimeUnit::Nanosecond, TimeUnit::Nanosecond) => 1.0,
    }
}

/// Parses `value` to `Option<i64>` consistent with the Arrow's definition of timestamp with timezone.
///
/// `tz` must be built from [`parse_offset`] or `chrono-tz`.
/// Returns in scale `tz` of `TimeUnit`.
#[inline]
pub fn utf8_to_timestamp_scalar(
    value: &str,
    fmt: &str,
    tz: &TimeZone,
    tu: &TimeUnit,
) -> Option<i64> {
    // "%+" mirrors chrono's combined ISO 8601 / RFC 3339 format specifier,
    // which jiff's strtime engine does not implement as a directive; parse
    // via jiff's native ISO 8601 timestamp parser instead, which (like
    // chrono's `%+`) honors the string's own embedded offset or "Z".
    let ts = if fmt == "%+" {
        value.parse::<Timestamp>().ok()?
    } else {
        let tm = jiff::fmt::strtime::BrokenDownTime::parse(fmt, value).ok()?;
        match tm.to_timestamp() {
            Ok(ts) => ts,
            // The format string carried no offset/zone info (e.g. no `%z`),
            // so reinterpret the parsed wall-clock reading as belonging to `tz`.
            Err(_) => {
                let naive = tm.to_datetime().ok()?;
                tz.to_ambiguous_zoned(naive).compatible().ok()?.timestamp()
            },
        }
    };
    Some(match tu {
        TimeUnit::Second => ts.as_second(),
        TimeUnit::Millisecond => ts.as_millisecond(),
        TimeUnit::Microsecond => ts.as_microsecond(),
        TimeUnit::Nanosecond => i64::try_from(ts.as_nanosecond()).ok()?,
    })
}

/// Parses an offset of the form `"+WX:YZ"` or `"UTC"` into a fixed-offset [`TimeZone`].
/// # Errors
/// If the offset is not in any of the allowed forms.
pub fn parse_offset(offset: &str) -> PolarsResult<TimeZone> {
    if offset == "UTC" {
        return Ok(TimeZone::UTC);
    }
    static ERR_MSG: &str = "timezone offset must be of the form [-]00:00";

    let mut a = offset.split(':');
    let first: &str = a
        .next()
        .ok_or_else(|| polars_err!(InvalidOperation: ERR_MSG))?;
    let last = a
        .next()
        .ok_or_else(|| polars_err!(InvalidOperation: ERR_MSG))?;
    let hours: i32 = first
        .parse()
        .map_err(|_| polars_err!(InvalidOperation: ERR_MSG))?;
    let minutes: i32 = last
        .parse()
        .map_err(|_| polars_err!(InvalidOperation: ERR_MSG))?;

    let offset = jiff::tz::Offset::from_seconds(hours * 60 * 60 + minutes * 60)
        .map_err(|_| polars_err!(InvalidOperation: "timezone offset out of bounds"))?;
    Ok(offset.to_time_zone())
}

/// Parses `value` to a [`TimeZone`] with the Arrow's definition of timestamp with a timezone.
#[cfg(feature = "chrono-tz")]
#[cfg_attr(docsrs, doc(cfg(feature = "chrono-tz")))]
pub fn parse_offset_tz(timezone: &str) -> PolarsResult<TimeZone> {
    TimeZone::get(timezone)
        .map_err(|_| polars_err!(InvalidOperation: "timezone \"{timezone}\" cannot be parsed"))
}
