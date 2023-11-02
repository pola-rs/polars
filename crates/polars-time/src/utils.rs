#[cfg(feature = "timezones")]
use arrow::legacy::kernels::{convert_to_naive_local, Ambiguous};
#[cfg(feature = "timezones")]
use arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use arrow::temporal_conversions::{
    timestamp_ms_to_datetime, timestamp_ns_to_datetime, timestamp_us_to_datetime,
};
#[cfg(feature = "timezones")]
use chrono::NaiveDateTime;
#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(feature = "timezones")]
use polars_core::prelude::{PolarsResult, TimeUnit};

#[cfg(feature = "timezones")]
pub(crate) fn localize_datetime(
    ndt: NaiveDateTime,
    tz: &Tz,
    ambiguous: Ambiguous,
) -> PolarsResult<NaiveDateTime> {
    // e.g. '2021-01-01 03:00' -> '2021-01-01 03:00CDT'
    convert_to_naive_local(&chrono_tz::UTC, tz, ndt, ambiguous)
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_datetime(ndt: NaiveDateTime, tz: &Tz) -> NaiveDateTime {
    // e.g. '2021-01-01 03:00CDT' -> '2021-01-01 03:00'
    tz.from_utc_datetime(&ndt).naive_local()
}

#[cfg(feature = "timezones")]
pub(crate) fn localize_timestamp(timestamp: i64, tu: TimeUnit, tz: Tz) -> PolarsResult<i64> {
    match tu {
        TimeUnit::Nanoseconds => Ok(convert_to_naive_local(
            &chrono_tz::UTC,
            &tz,
            timestamp_ns_to_datetime(timestamp),
            Ambiguous::Raise,
        )?
        .timestamp_nanos_opt()
        .unwrap()),
        TimeUnit::Microseconds => Ok(convert_to_naive_local(
            &chrono_tz::UTC,
            &tz,
            timestamp_us_to_datetime(timestamp),
            Ambiguous::Raise,
        )?
        .timestamp_micros()),
        TimeUnit::Milliseconds => Ok(convert_to_naive_local(
            &chrono_tz::UTC,
            &tz,
            timestamp_ms_to_datetime(timestamp),
            Ambiguous::Raise,
        )?
        .timestamp_millis()),
    }
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_timestamp(timestamp: i64, tu: TimeUnit, tz: Tz) -> i64 {
    match tu {
        TimeUnit::Nanoseconds => unlocalize_datetime(timestamp_ns_to_datetime(timestamp), &tz)
            .timestamp_nanos_opt()
            .unwrap(),
        TimeUnit::Microseconds => {
            unlocalize_datetime(timestamp_us_to_datetime(timestamp), &tz).timestamp_micros()
        },
        TimeUnit::Milliseconds => {
            unlocalize_datetime(timestamp_ms_to_datetime(timestamp), &tz).timestamp_millis()
        },
    }
}
