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

/// The instant `count` sub-second units after the epoch, where `per_day` of them make up a day and
/// `per_second` of them a second.
///
/// The day the count falls in and the time of day within it are worked out separately, which is
/// what makes this cheap: adding a [`TimeDelta`] onto the epoch datetime instead carries the
/// general date-and-time arithmetic, and a field extraction over a column spends some 2.4x the
/// instructions per element on it.
#[inline]
fn timestamp_to_datetime_opt(count: i64, per_day: i64, per_second: i64) -> Option<NaiveDateTime> {
    let days = i32::try_from(count.div_euclid(per_day)).ok()?;
    // A whole day of sub-second units need not fit in a `u32`; the seconds and the sub-second
    // remainder that come out of it do.
    let rem = count.rem_euclid(per_day);

    let date = date32_to_date_opt(days)?;
    let time = NaiveTime::from_num_seconds_from_midnight_opt(
        (rem / per_second) as u32,
        (rem % per_second) as u32 * (NANOSECONDS / per_second) as u32,
    )?;

    Some(date.and_time(time))
}

/// converts a `i32` representing a `date32` to [`NaiveDateTime`]
#[inline]
pub fn date32_to_datetime_opt(v: i32) -> Option<NaiveDateTime> {
    // A date is midnight of the day it names, so there is no time of day to work out.
    Some(date32_to_date_opt(v)?.and_time(NaiveTime::MIN))
}

/// converts a `i32` representing a `date32` to [`NaiveDate`]
#[inline]
pub fn date32_to_date(days: i32) -> NaiveDate {
    date32_to_date_opt(days).expect("out-of-range date")
}

/// converts a `i32` representing a `date32` to [`NaiveDate`]
#[inline]
pub fn date32_to_date_opt(days: i32) -> Option<NaiveDate> {
    NaiveDate::from_num_days_from_ce_opt(EPOCH_DAYS_FROM_CE.checked_add(days)?)
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
    timestamp_to_datetime_opt(v, MILLISECONDS_IN_DAY, MILLISECONDS)
}

/// converts a `i64` representing a `timestamp(us)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_us_to_datetime(v: i64) -> NaiveDateTime {
    timestamp_us_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(us)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_us_to_datetime_opt(v: i64) -> Option<NaiveDateTime> {
    timestamp_to_datetime_opt(v, MICROSECONDS_IN_DAY, MICROSECONDS)
}

/// converts a `i64` representing a `timestamp(ns)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ns_to_datetime(v: i64) -> NaiveDateTime {
    timestamp_ns_to_datetime_opt(v).expect("invalid or out-of-range datetime")
}

/// converts a `i64` representing a `timestamp(ns)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ns_to_datetime_opt(v: i64) -> Option<NaiveDateTime> {
    timestamp_to_datetime_opt(v, NANOSECONDS_IN_DAY, NANOSECONDS)
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
mod test {
    use super::*;

    /// The `TimeDelta` arithmetic the conversions below replaced, kept as the reference they are
    /// checked against: they read the day and the time of day separately instead, which is
    /// cheaper but has its own boundaries to get right.
    fn reference(count: i64, per_second: i64) -> Option<NaiveDateTime> {
        let delta = TimeDelta::new(
            count.div_euclid(per_second),
            (count.rem_euclid(per_second) * (NANOSECONDS / per_second)) as u32,
        )?;
        unix_epoch().checked_add_signed(delta)
    }

    /// Both `i64` extremes, both edges of the representable window, a stride across the whole
    /// range, and every count around the epoch and the day boundaries either side of it — where
    /// the euclidean split changes sign.
    fn check_unit(
        name: &str,
        convert: fn(i64) -> Option<NaiveDateTime>,
        per_day: i64,
        any_out_of_range: bool,
    ) {
        let per_second = per_day / SECONDS_IN_DAY;
        let mut out_of_range = 0;
        let mut check = |count: i64| {
            assert_eq!(
                convert(count),
                reference(count, per_second),
                "{name} conversion disagrees at {count}"
            );
            out_of_range += usize::from(convert(count).is_none());
        };

        for count in (i64::MIN..i64::MIN + 500).chain(i64::MAX - 500..i64::MAX) {
            check(count);
        }
        check(i64::MAX);

        // The edges of the window, found by bisecting on whether the count is representable.
        for (mut inside, mut outside) in [(0i64, i64::MIN), (0i64, i64::MAX)] {
            while inside.abs_diff(outside) > 1 {
                let middle = inside + (outside - inside) / 2;
                if convert(middle).is_some() {
                    inside = middle;
                } else {
                    outside = middle;
                }
            }
            for count in inside.saturating_sub(500)..inside.saturating_add(500) {
                check(count);
            }
        }

        let mut count = i64::MIN;
        while count < i64::MAX - (1 << 45) {
            check(count);
            count += 1 << 45;
        }
        for count in -20_000i64..20_000 {
            check(count);
        }
        for day in [-2i64, -1, 1, 2] {
            for offset in -500..500 {
                check(day * per_day + offset);
            }
        }

        // Every `i64` nanosecond count lands inside the datetime range, so `ns` has no `None`
        // boundary to check; the other two do, and it is the interesting part of the split.
        assert_eq!(
            out_of_range > 0,
            any_out_of_range,
            "{name}: {out_of_range} counts were out of range"
        );
    }

    #[test]
    fn timestamp_conversions_agree_with_adding_a_time_delta() {
        check_unit(
            "ms",
            timestamp_ms_to_datetime_opt,
            MILLISECONDS_IN_DAY,
            true,
        );
        check_unit(
            "us",
            timestamp_us_to_datetime_opt,
            MICROSECONDS_IN_DAY,
            true,
        );
        check_unit(
            "ns",
            timestamp_ns_to_datetime_opt,
            NANOSECONDS_IN_DAY,
            false,
        );
    }

    /// A date is midnight of the day it names, whichever way round it is worked out — including at
    /// the `i32` extremes, where the day count added to the epoch overflows.
    #[test]
    fn date32_conversions_agree_with_adding_a_time_delta() {
        let reference =
            |days: i32| unix_epoch().checked_add_signed(TimeDelta::try_days(days.into())?);

        let mut out_of_range = 0;
        for days in (i32::MIN..i32::MIN + 1_000)
            .chain(-500_000..500_000)
            .chain(i32::MAX - 1_000..i32::MAX)
            .chain([i32::MAX])
        {
            assert_eq!(
                date32_to_datetime_opt(days),
                reference(days),
                "date32 conversion disagrees at {days}"
            );
            assert_eq!(
                date32_to_date_opt(days),
                reference(days).map(|instant| instant.date()),
                "date32_to_date_opt disagrees at {days}"
            );
            out_of_range += usize::from(reference(days).is_none());
        }
        assert!(out_of_range > 0);
    }
}
