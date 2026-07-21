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

/// Convert time-units-since-epoch to a more structured object.
pub fn timestamp_to_naive_datetime(since_epoch: i64, time_unit: TimeUnit) -> NaiveDateTime {
    TimeZone::UTC.to_datetime(timestamp_to_timestamp(since_epoch, time_unit))
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
        let parsed_tz = match TimeZone::get(time_zone.as_str()) {
            Ok(tz) => tz,
            Err(_) => arrow::temporal_conversions::parse_offset(time_zone.as_str())
                .map_err(|_| PyPolarsErr::Other(format!("Could not parse timezone: {time_zone}")))?,
        };
        let ts = timestamp_to_timestamp(v, tu);
        ts.to_zoned(parsed_tz).into_bound_py_any(py)
    } else {
        timestamp_to_naive_datetime(v, tu).into_bound_py_any(py)
    }
}
