use arrow::array::{PrimitiveArray, Utf8ViewArray};
use arrow::datatypes::{ArrowDataType, TimeUnit};
pub use arrow::temporal_conversions::{
    EPOCH_DAYS_FROM_CE, MICROSECONDS, MICROSECONDS_IN_DAY, MILLISECONDS, MILLISECONDS_IN_DAY,
    NANOSECONDS, NANOSECONDS_IN_DAY, SECONDS_IN_DAY, utf8_to_timestamp_scalar,
};
use arrow::temporal_conversions::{date_to_date32_opt, parse_offset, parse_offset_tz};
use jiff::civil::{Date as NaiveDate, DateTime as NaiveDateTime, Time as NaiveTime};
use jiff::tz::TimeZone;
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

/// Get the time unit as a multiple of a second
pub const fn time_unit_multiple(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => MILLISECONDS,
        TimeUnit::Microsecond => MICROSECONDS,
        TimeUnit::Nanosecond => NANOSECONDS,
    }
}

fn named_tz_utf_to_timestamp(
    array: &Utf8ViewArray,
    fmt: &str,
    time_zone: PlSmallStr,
    time_unit: TimeUnit,
) -> PolarsResult<PrimitiveArray<i64>> {
    let tz = parse_offset_tz(time_zone.as_str())?;
    Ok(utf8view_to_timestamp_impl(
        array, fmt, time_zone, tz, time_unit,
    ))
}

fn utf8view_to_timestamp_impl(
    array: &Utf8ViewArray,
    fmt: &str,
    time_zone: PlSmallStr,
    tz: TimeZone,
    time_unit: TimeUnit,
) -> PrimitiveArray<i64> {
    let iter = array
        .iter()
        .map(|x| x.and_then(|x| utf8_to_timestamp_scalar(x, fmt, &tz, &time_unit)));

    PrimitiveArray::from_trusted_len_iter(iter)
        .to(ArrowDataType::Timestamp(time_unit, Some(time_zone)))
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
    time_zone: PlSmallStr,
    time_unit: TimeUnit,
) -> PolarsResult<PrimitiveArray<i64>> {
    let tz = parse_offset(time_zone.as_str());

    if let Ok(tz) = tz {
        Ok(utf8view_to_timestamp_impl(
            array, fmt, time_zone, tz, time_unit,
        ))
    } else {
        named_tz_utf_to_timestamp(array, fmt, time_zone, time_unit)
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
        NaiveDateTime::strptime(fmt, value).ok().or_else(|| {
            // This is only ever called with the fixed RFC3339 format (a
            // trailing `%:z`) for the "naive" cast path - jiff requires an
            // exact structural match, so a value with no offset at all
            // (genuinely naive input) needs the offset directive dropped
            // and the parse retried, matching this cast's contract of
            // accepting offset-less input.
            NaiveDateTime::strptime(fmt.strip_suffix("%:z")?, value).ok()
        })?
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
