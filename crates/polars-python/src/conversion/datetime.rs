//! Utilities for converting dates, times, datetimes, and so on.

use std::str::FromStr;

use chrono::{DateTime, FixedOffset, NaiveDateTime, NaiveTime, TimeDelta, TimeZone as _};
use chrono_tz::Tz;
use polars::datatypes::TimeUnit;
use polars_core::datatypes::TimeZone;
use pyo3::{Bound, IntoPyObject, PyAny, PyResult, Python};

use crate::error::PyPolarsErr;

pub fn elapsed_offset_to_timedelta(elapsed: i64, time_unit: TimeUnit) -> TimeDelta {
    let (in_second, nano_multiplier) = match time_unit {
        TimeUnit::Nanoseconds => (1_000_000_000, 1),
        TimeUnit::Microseconds => (1_000_000, 1_000),
        TimeUnit::Milliseconds => (1_000, 1_000_000),
    };
    let mut elapsed_sec = elapsed / in_second;
    let mut elapsed_nanos = nano_multiplier * (elapsed % in_second);
    if elapsed_nanos < 0 {
        // TimeDelta expects nanos to always be positive.
        elapsed_sec -= 1;
        elapsed_nanos += 1_000_000_000;
    }
    TimeDelta::new(elapsed_sec, elapsed_nanos as u32).unwrap()
}

/// Convert time-units-since-epoch to a more structured object.
pub fn timestamp_to_naive_datetime(since_epoch: i64, time_unit: TimeUnit) -> NaiveDateTime {
    NaiveDateTime::UNIX_EPOCH + elapsed_offset_to_timedelta(since_epoch, time_unit)
}

/// Convert nanoseconds-since-midnight to a more structured object.
pub fn nanos_since_midnight_to_naivetime(nanos_since_midnight: i64) -> NaiveTime {
    NaiveTime::from_hms_opt(0, 0, 0).unwrap()
        + elapsed_offset_to_timedelta(nanos_since_midnight, TimeUnit::Nanoseconds)
}

pub fn datetime_to_py_object<'py>(
    py: Python<'py>,
    v: i64,
    tu: TimeUnit,
    tz: Option<&TimeZone>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(time_zone) = tz {
        if let Ok(tz) = Tz::from_str(time_zone) {
            let utc_datetime = DateTime::UNIX_EPOCH + elapsed_offset_to_timedelta(v, tu);
            let datetime = utc_datetime.with_timezone(&tz);
            datetime.into_pyobject(py)
        } else if let Ok(tz) = FixedOffset::from_str(time_zone) {
            let naive_datetime = timestamp_to_naive_datetime(v, tu);
            let datetime = tz.from_utc_datetime(&naive_datetime);
            datetime.into_pyobject(py)
        } else {
            Err(PyPolarsErr::Other(format!("Could not parse timezone: {time_zone}")).into())
        }
    } else {
        timestamp_to_naive_datetime(v, tu).into_pyobject(py)
    }
}
