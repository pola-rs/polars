//! Conversion methods for dates and times.

use chrono::format::{Parsed, StrftimeItems, parse};
use chrono::{DateTime, Duration, FixedOffset, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta};
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
fn unix_epoch() -> NaiveDateTime {
    DateTime::UNIX_EPOCH.naive_utc()
}

/// converts a `i32` representing a `date32` to [`NaiveDateTime`]
#[inline]
pub fn date32_to_datetime(v: i32) -> NaiveDateTime {
    date32_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i32` representing a `date32` to [`NaiveDateTime`]
#[inline]
pub fn date32_to_datetime_opt(v: i32) -> Option<NaiveDateTime> {
    let delta = TimeDelta::try_days(v.into())?;
    unix_epoch().checked_add_signed(delta)
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
        .and_then(|delta| unix_epoch().checked_add_signed(delta))
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
    unix_epoch().checked_add_signed(delta)
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
    unix_epoch().checked_add_signed(delta)
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
    unix_epoch().checked_add_signed(delta)
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

/// Parses a datetime string according to format `fmt`.
///
/// When `fmt == "%+"` (the ISO-8601 format used in JSON parsing), this also
/// attempts common ISO-8601 fallback patterns to handle timestamps lacking seconds
/// (e.g., `"2020-01-01T12:34+05:00"` or `"2020-01-01T12:34"`).
///
/// Returns a [`NaiveDateTime`] normalized to UTC if an explicit timezone offset was present,
/// or preserving the naive wall-clock value if no offset was present.
#[inline]
pub fn parse_iso8601_datetime_components(value: &str, fmt: &str) -> Option<NaiveDateTime> {
    let mut parsed = Parsed::new();
    let fmt_items = StrftimeItems::new(fmt);
    let mut r = parse(&mut parsed, value, fmt_items).ok();
    if r.is_none() && fmt == "%+" {
        for fallback in [
            "%Y-%m-%dT%H:%M%#z",
            "%Y-%m-%dT%H:%M:%S%.f%#z",
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S%.f%#z",
            "%Y-%m-%d %H:%M%#z",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M",
        ] {
            parsed = Parsed::new();
            if parse(&mut parsed, value, StrftimeItems::new(fallback)).is_ok() {
                r = Some(());
                break;
            }
        }
    }
    if r.is_some() {
        parsed
            .to_datetime()
            .map(|dt| dt.naive_utc())
            .or_else(|_| parsed.to_naive_datetime_with_offset(0))
            .ok()
    } else {
        None
    }
}

/// Parses `value` to `Option<i64>` consistent with the Arrow's definition of timestamp with timezone.
///
/// `tz` must be built from `timezone` (either via [`parse_offset`] or `chrono-tz`).
/// Returns in scale `tu` of `TimeUnit`.
#[inline]
pub fn utf8_to_timestamp_scalar<T: chrono::TimeZone>(
    value: &str,
    fmt: &str,
    tz: &T,
    tu: &TimeUnit,
) -> Option<i64> {
    let ndt = parse_iso8601_datetime_components(value, fmt)?;
    let dt = tz.from_utc_datetime(&ndt);
    Some(match tu {
        TimeUnit::Second => dt.timestamp(),
        TimeUnit::Millisecond => dt.timestamp_millis(),
        TimeUnit::Microsecond => dt.timestamp_micros(),
        TimeUnit::Nanosecond => dt.timestamp_nanos_opt().unwrap(),
    })
}

/// Parses an offset of the form `"+WX:YZ"` or `"UTC"` into [`FixedOffset`].
/// # Errors
/// If the offset is not in any of the allowed forms.
pub fn parse_offset(offset: &str) -> PolarsResult<FixedOffset> {
    if offset == "UTC" {
        return Ok(FixedOffset::east_opt(0).expect("FixedOffset::east out of bounds"));
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

    Ok(FixedOffset::east_opt(hours * 60 * 60 + minutes * 60)
        .expect("FixedOffset::east out of bounds"))
}

/// Parses `value` to a [`chrono_tz::Tz`] with the Arrow's definition of timestamp with a timezone.
#[cfg(feature = "chrono-tz")]
#[cfg_attr(docsrs, doc(cfg(feature = "chrono-tz")))]
pub fn parse_offset_tz(timezone: &str) -> PolarsResult<chrono_tz::Tz> {
    timezone
        .parse::<chrono_tz::Tz>()
        .map_err(|_| polars_err!(InvalidOperation: "timezone \"{timezone}\" cannot be parsed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_iso8601_datetime_components() {
        // 1. Issue #29262: different offsets resolve to correct UTC physical instants
        let dt_p5 = parse_iso8601_datetime_components("2020-01-01T12:34+05:00", "%+").unwrap();
        let dt_p6 = parse_iso8601_datetime_components("2020-01-01T12:34+06:00", "%+").unwrap();
        let dt_utc = parse_iso8601_datetime_components("2020-01-01T07:34Z", "%+").unwrap();
        assert_ne!(dt_p5, dt_p6);
        assert_eq!(dt_p5, dt_utc);
        assert_eq!(
            dt_p5,
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(7, 34, 0)
                .unwrap()
        );
        assert_eq!(
            dt_p6,
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(6, 34, 0)
                .unwrap()
        );

        // 2. Seconds vs minute precision
        let dt_p5_sec =
            parse_iso8601_datetime_components("2020-01-01T12:34:00+05:00", "%+").unwrap();
        assert_eq!(dt_p5, dt_p5_sec);

        // 3. Negative offset
        let dt_neg = parse_iso8601_datetime_components("2020-01-01T12:34-05:00", "%+").unwrap();
        assert_eq!(
            dt_neg,
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(17, 34, 0)
                .unwrap()
        );

        // 4. Fractional-hour offsets
        let dt_530 = parse_iso8601_datetime_components("2020-01-01T12:34+05:30", "%+").unwrap();
        assert_eq!(
            dt_530,
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(7, 4, 0)
                .unwrap()
        );

        // 5. Date-crossing boundaries
        let dt_cross_back =
            parse_iso8601_datetime_components("2020-01-01T01:00+05:30", "%+").unwrap();
        assert_eq!(
            dt_cross_back,
            NaiveDate::from_ymd_opt(2019, 12, 31)
                .unwrap()
                .and_hms_opt(19, 30, 0)
                .unwrap()
        );
        let dt_cross_fwd =
            parse_iso8601_datetime_components("2020-01-01T23:30-05:00", "%+").unwrap();
        assert_eq!(
            dt_cross_fwd,
            NaiveDate::from_ymd_opt(2020, 1, 2)
                .unwrap()
                .and_hms_opt(4, 30, 0)
                .unwrap()
        );

        // 6. Naive timestamps preserve wall-clock
        let dt_naive = parse_iso8601_datetime_components("2020-01-01T12:34", "%+").unwrap();
        let dt_naive_sec = parse_iso8601_datetime_components("2020-01-01T12:34:00", "%+").unwrap();
        let dt_naive_space = parse_iso8601_datetime_components("2020-01-01 12:34", "%+").unwrap();
        let exp_naive = NaiveDate::from_ymd_opt(2020, 1, 1)
            .unwrap()
            .and_hms_opt(12, 34, 0)
            .unwrap();
        assert_eq!(dt_naive, exp_naive);
        assert_eq!(dt_naive_sec, exp_naive);
        assert_eq!(dt_naive_space, exp_naive);

        // 7. Invalid strings return None
        assert_eq!(parse_iso8601_datetime_components("invalid", "%+"), None);
        assert_eq!(
            parse_iso8601_datetime_components("2020-01-01T12:34GARBAGE", "%+"),
            None
        );
        assert_eq!(parse_iso8601_datetime_components("2020-01-01", "%+"), None);
    }

    #[test]
    fn test_utf8_to_timestamp_scalar() {
        let tu = TimeUnit::Microsecond;
        let utc_tz = FixedOffset::east_opt(0).unwrap();

        let ts_p5 = utf8_to_timestamp_scalar("2020-01-01T12:34+05:00", "%+", &utc_tz, &tu).unwrap();
        let ts_p6 = utf8_to_timestamp_scalar("2020-01-01T12:34+06:00", "%+", &utc_tz, &tu).unwrap();
        let ts_utc = utf8_to_timestamp_scalar("2020-01-01T07:34Z", "%+", &utc_tz, &tu).unwrap();

        assert_ne!(ts_p5, ts_p6);
        assert_eq!(ts_p5, ts_utc);
        assert_eq!(ts_p5 - ts_p6, 3600 * 1_000_000);
    }
}
