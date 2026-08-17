//! Utilities for converting dates, times, datetimes, and so on.

use jiff::civil::{DateTime as NaiveDateTime, Time as NaiveTime};
use jiff::tz::TimeZone;
use jiff::{SignedDuration as TimeDelta, Timestamp};
use polars::datatypes::TimeUnit;
use polars_core::datatypes::TimeZone as PlTimeZone;
use pyo3::{Bound, IntoPyObjectExt, PyAny, PyResult, Python};

use crate::error::PyPolarsErr;

pub fn elapsed_offset_to_timedelta(elapsed: i64, time_unit: TimeUnit) -> TimeDelta {
    match time_unit {
        TimeUnit::Nanoseconds => TimeDelta::from_nanos(elapsed),
        TimeUnit::Microseconds => TimeDelta::from_micros(elapsed),
        TimeUnit::Milliseconds => TimeDelta::from_millis(elapsed),
    }
}

fn timestamp_to_timestamp(since_epoch: i64, time_unit: TimeUnit) -> Timestamp {
    match time_unit {
        TimeUnit::Nanoseconds => Timestamp::from_nanosecond(i128::from(since_epoch)),
        TimeUnit::Microseconds => Timestamp::from_microsecond(since_epoch),
        TimeUnit::Milliseconds => Timestamp::from_millisecond(since_epoch),
    }
    .expect("timestamp out-of-range")
}

fn timestamp_to_timestamp_opt(since_epoch: i64, time_unit: TimeUnit) -> Option<Timestamp> {
    match time_unit {
        TimeUnit::Nanoseconds => Timestamp::from_nanosecond(i128::from(since_epoch)),
        TimeUnit::Microseconds => Timestamp::from_microsecond(since_epoch),
        TimeUnit::Milliseconds => Timestamp::from_millisecond(since_epoch),
    }
    .ok()
}

/// Convert time-units-since-epoch to a more structured object.
///
/// Computed via calendar arithmetic rather than through `Timestamp` (see
/// `arrow::temporal_conversions::epoch_nanos_to_datetime_opt`), since
/// `Timestamp`'s representable range is narrower than `DateTime`'s and would
/// otherwise reject legitimate values near the -9999/9999 year boundary.
pub fn timestamp_to_naive_datetime(since_epoch: i64, time_unit: TimeUnit) -> NaiveDateTime {
    let nanos: i128 = match time_unit {
        TimeUnit::Nanoseconds => i128::from(since_epoch),
        TimeUnit::Microseconds => i128::from(since_epoch) * 1_000,
        TimeUnit::Milliseconds => i128::from(since_epoch) * 1_000_000,
    };
    arrow::temporal_conversions::epoch_nanos_to_datetime_opt(nanos).expect("datetime out-of-range")
}

/// Convert nanoseconds-since-midnight to a more structured object.
pub fn nanos_since_midnight_to_naivetime(nanos_since_midnight: i64) -> NaiveTime {
    NaiveTime::midnight()
        .checked_add(jiff::Span::new().nanoseconds(nanos_since_midnight))
        .expect("time out-of-range")
}

pub fn datetime_to_py_object<'py>(
    py: Python<'py>,
    v: i64,
    tu: TimeUnit,
    tz: Option<&PlTimeZone>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(time_zone) = tz {
        match TimeZone::get(time_zone.as_str()) {
            Ok(parsed_tz) => {
                // pyo3's `IntoPyObject for Zoned`/`TimeZone` special-cases
                // `TimeZone::UTC` to always produce Python's fixed
                // `datetime.timezone.utc`, even when the zone was obtained
                // by IANA name (e.g. "UTC" or "Etc/UTC") - build the object
                // manually via an explicit `zoneinfo.ZoneInfo` lookup
                // instead, so a named time zone always round-trips to a
                // `ZoneInfo`, matching this dtype's contract.
                let (local, offset) = match timestamp_to_timestamp_opt(v, tu) {
                    Some(ts) => {
                        let zoned = ts.to_zoned(parsed_tz.clone());
                        (zoned.datetime(), zoned.offset())
                    },
                    None => {
                        // Extreme tail beyond `Timestamp`'s (narrower than
                        // `DateTime`'s) range - approximate the offset using
                        // the UTC civil reading as the tz-rule lookup key
                        // instead of the true instant. Exact for zones
                        // without DST; for DST zones this can only be off
                        // by one DST delta, and only within this narrow
                        // out-of-`Timestamp`-range tail.
                        let naive_utc = timestamp_to_naive_datetime(v, tu);
                        let offset = match parsed_tz.to_ambiguous_zoned(naive_utc).offset() {
                            jiff::tz::AmbiguousOffset::Unambiguous { offset } => offset,
                            jiff::tz::AmbiguousOffset::Fold { before, .. } => before,
                            jiff::tz::AmbiguousOffset::Gap { before, .. } => before,
                        };
                        let local = naive_utc
                            .checked_add(jiff::Span::new().seconds(i64::from(offset.seconds())))
                            .map_err(|e| PyPolarsErr::Other(e.to_string()))?;
                        (local, offset)
                    },
                };
                // Determine Python's `fold`: true iff `local` is the *later*
                // of two repeated wall-clock readings during a DST fold
                // (i.e. our resolved `offset` matches the fold's `after`
                // side) - without this, the fold information is lost and a
                // round-tripped ambiguous datetime becomes indistinguishable
                // from its earlier twin.
                let fold = matches!(
                    parsed_tz.to_ambiguous_zoned(local).offset(),
                    jiff::tz::AmbiguousOffset::Fold { after, .. } if after == offset
                );
                let py_tz = pyo3::types::PyTzInfo::timezone(py, time_zone.as_str())?;
                pyo3::types::PyDateTime::new_with_fold(
                    py,
                    i32::from(local.year()),
                    local.month() as u8,
                    local.day() as u8,
                    local.hour() as u8,
                    local.minute() as u8,
                    local.second() as u8,
                    (local.subsec_nanosecond() / 1_000) as u32,
                    Some(&py_tz),
                    fold,
                )
                .map(|dt| dt.into_any())
            },
            Err(_) => {
                let parsed_tz = arrow::temporal_conversions::parse_offset(time_zone.as_str())
                    .map_err(|_| {
                        PyPolarsErr::Other(format!("Could not parse timezone: {time_zone}"))
                    })?;
                let ts = timestamp_to_timestamp(v, tu);
                ts.to_zoned(parsed_tz).into_bound_py_any(py)
            },
        }
    } else {
        timestamp_to_naive_datetime(v, tu).into_bound_py_any(py)
    }
}
