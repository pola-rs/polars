#[cfg(feature = "timezones")]
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(feature = "timezones")]
use chrono::{LocalResult, NaiveDateTime};
#[cfg(feature = "timezones")]
use polars_arrow::time_zone::Tz;
#[cfg(feature = "timezones")]
use polars_core::prelude::{polars_bail, PolarsResult, TimeUnit};

#[cfg(feature = "timezones")]
pub(crate) fn localize_datetime(
    ndt: NaiveDateTime,
    tz: &Tz,
    use_earliest: Option<bool>,
) -> PolarsResult<NaiveDateTime> {
    // e.g. '2021-01-01 03:00' -> '2021-01-01 03:00CDT'
    match tz.from_local_datetime(&ndt) {
        LocalResult::Single(tz) => Ok(tz.naive_utc()),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match use_earliest {
            Some(true) => Ok(dt_earliest.naive_utc()),
            Some(false) => Ok(dt_latest.naive_utc()),
            None => polars_bail!(ComputeError:
                format!("datetime '{}' is ambiguous in time zone '{}'. \
                    Please use `use_earliest` to tell how it should be localized. \
                    If you got here from a function which doesn't have a `use_earliest` argument, \
                    please open an issue at https://github.com/pola-rs/polars/issues.", ndt, tz)
            ),
        },
        LocalResult::None => {
            polars_bail!(
                ComputeError: format!("datetime '{}' is non-existent in time zone '{}'. Non-existent datetimes are not yet supported", ndt, tz)
            )
        }
    }
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_datetime(ndt: NaiveDateTime, tz: &Tz) -> NaiveDateTime {
    // e.g. '2021-01-01 03:00CDT' -> '2021-01-01 03:00'
    tz.from_utc_datetime(&ndt).naive_local()
}

#[cfg(feature = "timezones")]
pub(crate) fn localize_timestamp(timestamp: i64, tu: TimeUnit, tz: Tz) -> PolarsResult<i64> {
    match tu {
        TimeUnit::Nanoseconds => {
            Ok(
                localize_datetime(timestamp_ns_to_datetime(timestamp), &tz, None)?
                    .timestamp_nanos(),
            )
        }
        TimeUnit::Microseconds => {
            Ok(
                localize_datetime(timestamp_us_to_datetime(timestamp), &tz, None)?
                    .timestamp_micros(),
            )
        }
        TimeUnit::Milliseconds => {
            Ok(
                localize_datetime(timestamp_ms_to_datetime(timestamp), &tz, None)?
                    .timestamp_millis(),
            )
        }
    }
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_timestamp(timestamp: i64, tu: TimeUnit, tz: Tz) -> i64 {
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
