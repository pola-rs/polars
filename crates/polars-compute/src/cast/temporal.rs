use arrow::array::{PrimitiveArray, Utf8ViewArray};
use arrow::datatypes::{ArrowDataType, TimeUnit};
pub use arrow::temporal_conversions::{
    EPOCH_DAYS_FROM_CE, MICROSECONDS, MICROSECONDS_IN_DAY, MILLISECONDS, MILLISECONDS_IN_DAY,
    NANOSECONDS, NANOSECONDS_IN_DAY, SECONDS_IN_DAY,
};
use arrow::temporal_conversions::{parse_offset, parse_offset_tz};
use chrono::format::{Parsed, StrftimeItems};
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

fn chrono_tz_utf_to_timestamp(
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

fn utf8view_to_timestamp_impl<T: chrono::TimeZone>(
    array: &Utf8ViewArray,
    fmt: &str,
    time_zone: PlSmallStr,
    tz: T,
    time_unit: TimeUnit,
) -> PrimitiveArray<i64> {
    let iter = array
        .iter()
        .map(|x| x.and_then(|x| utf8_to_timestamp_scalar(x, fmt, &tz, &time_unit)));

    PrimitiveArray::from_trusted_len_iter(iter)
        .to(ArrowDataType::Timestamp(time_unit, Some(time_zone)))
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
