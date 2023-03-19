#[cfg(feature = "timezones")]
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::{LocalResult, NaiveDateTime, TimeZone as TimeZoneTrait};
#[cfg(feature = "timezones")]
use polars_core::prelude::{polars_bail, PolarsResult, TimeUnit};

#[cfg(feature = "timezones")]
pub(crate) fn localize_datetime(
    ndt: NaiveDateTime,
    tz: &impl TimeZoneTrait,
) -> PolarsResult<NaiveDateTime> {
    // e.g. '2021-01-01 03:00' -> '2021-01-01 03:00CDT'
    match tz.from_local_datetime(&ndt) {
        LocalResult::Single(tz) => Ok(tz.naive_utc()),
        LocalResult::Ambiguous(_, _) => {
            polars_bail!(ComputeError: "ambiguous timestamps are not (yet) supported")
        }
        LocalResult::None => {
            polars_bail!(ComputeError: "non-existent timestamps are not (yet) supported")
        }
    }
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_datetime(ndt: NaiveDateTime, tz: &impl TimeZoneTrait) -> NaiveDateTime {
    // e.g. '2021-01-01 03:00CDT' -> '2021-01-01 03:00'
    tz.from_utc_datetime(&ndt).naive_local()
}

#[cfg(feature = "timezones")]
pub(crate) fn localize_timestamp<T: TimeZoneTrait>(
    timestamp: i64,
    tu: TimeUnit,
    tz: T,
) -> PolarsResult<i64> {
    match tu {
        TimeUnit::Nanoseconds => {
            Ok(localize_datetime(timestamp_ns_to_datetime(timestamp), &tz)?.timestamp_nanos())
        }
        TimeUnit::Microseconds => {
            Ok(localize_datetime(timestamp_us_to_datetime(timestamp), &tz)?.timestamp_micros())
        }
        TimeUnit::Milliseconds => {
            Ok(localize_datetime(timestamp_ms_to_datetime(timestamp), &tz)?.timestamp_millis())
        }
    }
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_timestamp<T: TimeZoneTrait>(timestamp: i64, tu: TimeUnit, tz: T) -> i64 {
    match tu {
        TimeUnit::Nanoseconds => {
            unlocalize_datetime(timestamp_ns_to_datetime(timestamp), &tz).timestamp_nanos()
        }
        TimeUnit::Microseconds => {
            unlocalize_datetime(timestamp_us_to_datetime(timestamp), &tz).timestamp_micros()
        }
        TimeUnit::Milliseconds => {
            unlocalize_datetime(timestamp_ms_to_datetime(timestamp), &tz).timestamp_millis()
        }
    }
}
