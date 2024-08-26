//! Conversion methods for dates and times.

use chrono::format::{parse, Parsed, StrftimeItems};
use chrono::{DateTime, Duration, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta};
use polars_error::{polars_err, PolarsResult};

use crate::array::{PrimitiveArray, Utf8ViewArray};
use crate::datatypes::{ArrowDataType, TimeUnit};

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

/// converts a `i32` representing a `date32` to [`NaiveDateTime`]
#[inline]
pub fn date32_to_datetime(v: i32) -> NaiveDateTime {
    date32_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i32` representing a `date32` to [`NaiveDateTime`]
#[inline]
pub fn date32_to_datetime_opt(v: i32) -> Option<NaiveDateTime> {
    let delta = TimeDelta::try_days(v.into())?;
    NaiveDateTime::UNIX_EPOCH.checked_add_signed(delta)
}

/// converts a `i32` representing a `date32` to [`NaiveDate`]
#[inline]
pub fn date32_to_date(days: i32) -> NaiveDate {
    date32_to_date_opt(days).expect("out-of-range date")
}

/// converts a `i32` representing a `date32` to [`NaiveDate`]
#[inline]
pub fn date32_to_date_opt(days: i32) -> Option<NaiveDate> {
    NaiveDate::from_num_days_from_ce_opt(EPOCH_DAYS_FROM_CE + days)
}

/// converts a `i64` representing a `date64` to [`NaiveDateTime`]
#[inline]
pub fn date64_to_datetime(v: i64) -> NaiveDateTime {
    TimeDelta::try_milliseconds(v)
        .and_then(|delta| NaiveDateTime::UNIX_EPOCH.checked_add_signed(delta))
        .expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `date64` to [`NaiveDate`]
#[inline]
pub fn date64_to_date(milliseconds: i64) -> NaiveDate {
    date64_to_datetime(milliseconds).date()
}

/// converts a `i32` representing a `time32(s)` to [`NaiveTime`]
#[inline]
pub fn time32s_to_time(v: i32) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight_opt(v as u32, 0).expect("invalid time")
}

/// converts a `i64` representing a `duration(s)` to [`Duration`]
#[inline]
pub fn duration_s_to_duration(v: i64) -> Duration {
    Duration::try_seconds(v).expect("out-of-range duration")
}

/// converts a `i64` representing a `duration(ms)` to [`Duration`]
#[inline]
pub fn duration_ms_to_duration(v: i64) -> Duration {
    Duration::try_milliseconds(v).expect("out-of-range in duration conversion")
}

/// converts a `i64` representing a `duration(us)` to [`Duration`]
#[inline]
pub fn duration_us_to_duration(v: i64) -> Duration {
    Duration::microseconds(v)
}

/// converts a `i64` representing a `duration(ns)` to [`Duration`]
#[inline]
pub fn duration_ns_to_duration(v: i64) -> Duration {
    Duration::nanoseconds(v)
}

/// converts a `i32` representing a `time32(ms)` to [`NaiveTime`]
#[inline]
pub fn time32ms_to_time(v: i32) -> NaiveTime {
    let v = v as i64;
    let seconds = v / MILLISECONDS;

    let milli_to_nano = 1_000_000;
    let nano = (v - seconds * MILLISECONDS) * milli_to_nano;
    NaiveTime::from_num_seconds_from_midnight_opt(seconds as u32, nano as u32)
        .expect("invalid time")
}

/// converts a `i64` representing a `time64(us)` to [`NaiveTime`]
#[inline]
pub fn time64us_to_time(v: i64) -> NaiveTime {
    time64us_to_time_opt(v).expect("invalid time")
}

/// converts a `i64` representing a `time64(us)` to [`NaiveTime`]
#[inline]
pub fn time64us_to_time_opt(v: i64) -> Option<NaiveTime> {
    NaiveTime::from_num_seconds_from_midnight_opt(
        // extract seconds from microseconds
        (v / MICROSECONDS) as u32,
        // discard extracted seconds and convert microseconds to
        // nanoseconds
        (v % MICROSECONDS * MILLISECONDS) as u32,
    )
}

/// converts a `i64` representing a `time64(ns)` to [`NaiveTime`]
#[inline]
pub fn time64ns_to_time(v: i64) -> NaiveTime {
    time64ns_to_time_opt(v).expect("invalid time")
}

/// converts a `i64` representing a `time64(ns)` to [`NaiveTime`]
#[inline]
pub fn time64ns_to_time_opt(v: i64) -> Option<NaiveTime> {
    NaiveTime::from_num_seconds_from_midnight_opt(
        // extract seconds from nanoseconds
        (v / NANOSECONDS) as u32,
        // discard extracted seconds
        (v % NANOSECONDS) as u32,
    )
}

/// converts a `i64` representing a `timestamp(s)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_s_to_datetime(seconds: i64) -> NaiveDateTime {
    timestamp_s_to_datetime_opt(seconds).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(s)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_s_to_datetime_opt(seconds: i64) -> Option<NaiveDateTime> {
    Some(DateTime::from_timestamp(seconds, 0)?.naive_utc())
}

/// converts a `i64` representing a `timestamp(ms)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ms_to_datetime(v: i64) -> NaiveDateTime {
    timestamp_ms_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(ms)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ms_to_datetime_opt(v: i64) -> Option<NaiveDateTime> {
    let delta = TimeDelta::try_milliseconds(v)?;
    NaiveDateTime::UNIX_EPOCH.checked_add_signed(delta)
}

/// converts a `i64` representing a `timestamp(us)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_us_to_datetime(v: i64) -> NaiveDateTime {
    timestamp_us_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(us)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_us_to_datetime_opt(v: i64) -> Option<NaiveDateTime> {
    let delta = TimeDelta::microseconds(v);
    NaiveDateTime::UNIX_EPOCH.checked_add_signed(delta)
}

/// converts a `i64` representing a `timestamp(ns)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ns_to_datetime(v: i64) -> NaiveDateTime {
    timestamp_ns_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(ns)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ns_to_datetime_opt(v: i64) -> Option<NaiveDateTime> {
    let delta = TimeDelta::nanoseconds(v);
    NaiveDateTime::UNIX_EPOCH.checked_add_signed(delta)
}

/// Converts a timestamp in `time_unit` and `timezone` into [`chrono::DateTime`].
#[inline]
pub(crate) fn timestamp_to_naive_datetime(
    timestamp: i64,
    time_unit: TimeUnit,
) -> chrono::NaiveDateTime {
    match time_unit {
        TimeUnit::Second => timestamp_s_to_datetime(timestamp),
        TimeUnit::Millisecond => timestamp_ms_to_datetime(timestamp),
        TimeUnit::Microsecond => timestamp_us_to_datetime(timestamp),
        TimeUnit::Nanosecond => timestamp_ns_to_datetime(timestamp),
    }
}

/// Converts a timestamp in `time_unit` and `timezone` into [`chrono::DateTime`].
#[inline]
pub fn timestamp_to_datetime<T: chrono::TimeZone>(
    timestamp: i64,
    time_unit: TimeUnit,
    timezone: &T,
) -> chrono::DateTime<T> {
    timezone.from_utc_datetime(&timestamp_to_naive_datetime(timestamp, time_unit))
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

/// Parses an offset of the form `"+WX:YZ"` or `"UTC"` into [`FixedOffset`].
/// # Errors
/// If the offset is not in any of the allowed forms.
pub fn parse_offset(offset: &str) -> PolarsResult<FixedOffset> {
    if offset == "UTC" {
        return Ok(FixedOffset::east_opt(0).expect("FixedOffset::east out of bounds"));
    }
    let error = "timezone offset must be of the form [-]00:00";

    let mut a = offset.split(':');
    let first: &str = a
        .next()
        .ok_or_else(|| polars_err!(InvalidOperation: error))?;
    let last = a
        .next()
        .ok_or_else(|| polars_err!(InvalidOperation: error))?;
    let hours: i32 = first
        .parse()
        .map_err(|_| polars_err!(InvalidOperation: error))?;
    let minutes: i32 = last
        .parse()
        .map_err(|_| polars_err!(InvalidOperation: error))?;

    Ok(FixedOffset::east_opt(hours * 60 * 60 + minutes * 60)
        .expect("FixedOffset::east out of bounds"))
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
    let r = parse(&mut parsed, value, fmt).ok();
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
    parse(&mut parsed, value, fmt.clone()).ok();
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

fn utf8view_to_timestamp_impl<T: chrono::TimeZone>(
    array: &Utf8ViewArray,
    fmt: &str,
    time_zone: String,
    tz: T,
    time_unit: TimeUnit,
) -> PrimitiveArray<i64> {
    let iter = array
        .iter()
        .map(|x| x.and_then(|x| utf8_to_timestamp_scalar(x, fmt, &tz, &time_unit)));

    PrimitiveArray::from_trusted_len_iter(iter)
        .to(ArrowDataType::Timestamp(time_unit, Some(time_zone)))
}

/// Parses `value` to a [`chrono_tz::Tz`] with the Arrow's definition of timestamp with a timezone.
#[cfg(feature = "chrono-tz")]
#[cfg_attr(docsrs, doc(cfg(feature = "chrono-tz")))]
pub fn parse_offset_tz(timezone: &str) -> PolarsResult<chrono_tz::Tz> {
    timezone
        .parse::<chrono_tz::Tz>()
        .map_err(|_| polars_err!(InvalidOperation: "timezone \"{timezone}\" cannot be parsed"))
}

/// Get the time unit as a multiple of a second
pub const fn time_unit_multiple(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => MILLISECONDS,
        TimeUnit::Microsecond => MICROSECONDS,
        TimeUnit::Nanosecond => NANOSECONDS,
    }
}

#[cfg(feature = "chrono-tz")]
#[cfg_attr(docsrs, doc(cfg(feature = "chrono-tz")))]
fn chrono_tz_utf_to_timestamp(
    array: &Utf8ViewArray,
    fmt: &str,
    time_zone: String,
    time_unit: TimeUnit,
) -> PolarsResult<PrimitiveArray<i64>> {
    let tz = parse_offset_tz(&time_zone)?;
    Ok(utf8view_to_timestamp_impl(
        array, fmt, time_zone, tz, time_unit,
    ))
}

#[cfg(not(feature = "chrono-tz"))]
fn chrono_tz_utf_to_timestamp(
    _: &Utf8ViewArray,
    _: &str,
    timezone: String,
    _: TimeUnit,
) -> PolarsResult<PrimitiveArray<i64>> {
    panic!("timezone \"{timezone}\" cannot be parsed (feature chrono-tz is not active)")
}

/// Parses a [`Utf8Array`] to a timeozone-aware timestamp, i.e. [`PrimitiveArray<i64>`] with type `Timestamp(Nanosecond, Some(timezone))`.
///
/// # Implementation
///
/// * parsed values with timezone other than `timezone` are converted to `timezone`.
/// * parsed values without timezone are null. Use [`utf8_to_naive_timestamp`] to parse naive timezones.
/// * Null elements remain null; non-parsable elements are null.
///
/// The feature `"chrono-tz"` enables IANA and zoneinfo formats for `timezone`.
///
/// # Error
///
/// This function errors iff `timezone` is not parsable to an offset.
pub(crate) fn utf8view_to_timestamp(
    array: &Utf8ViewArray,
    fmt: &str,
    time_zone: String,
    time_unit: TimeUnit,
) -> PolarsResult<PrimitiveArray<i64>> {
    let tz = parse_offset(time_zone.as_str());

    if let Ok(tz) = tz {
        Ok(utf8view_to_timestamp_impl(
            array, fmt, time_zone, tz, time_unit,
        ))
    } else {
        chrono_tz_utf_to_timestamp(array, fmt, time_zone, time_unit)
    }
}

/// Parses a [`Utf8Array`] to naive timestamp, i.e.
/// [`PrimitiveArray<i64>`] with type `Timestamp(Nanosecond, None)`.
/// Timezones are ignored.
/// Null elements remain null; non-parsable elements are set to null.
pub(crate) fn utf8view_to_naive_timestamp(
    array: &Utf8ViewArray,
    fmt: &str,
    time_unit: TimeUnit,
) -> PrimitiveArray<i64> {
    let iter = array
        .iter()
        .map(|x| x.and_then(|x| utf8_to_naive_timestamp_scalar(x, fmt, &time_unit)));

    PrimitiveArray::from_trusted_len_iter(iter).to(ArrowDataType::Timestamp(time_unit, None))
}
