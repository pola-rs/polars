#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(feature = "timezones")]
use chrono::{LocalResult, NaiveDateTime};
#[cfg(feature = "timezones")]
use polars_arrow::time_zone::Tz;
#[cfg(feature = "timezones")]
use polars_core::prelude::{
    polars_bail, timestamp_to_naive_datetime_method, PolarsResult, TimeUnit,
};

#[cfg(feature = "timezones")]
pub(crate) fn localize_datetime(
    ndt: NaiveDateTime,
    tz: &Tz,
    ambiguous: &str,
) -> PolarsResult<NaiveDateTime> {
    // e.g. '2021-01-01 03:00' -> '2021-01-01 03:00CDT'
    match tz.from_local_datetime(&ndt) {
        LocalResult::Single(tz) => Ok(tz.naive_utc()),
        LocalResult::Ambiguous(dt_earliest, dt_latest) => match ambiguous {
            "earliest" => Ok(dt_earliest.naive_utc()),
            "latest" => Ok(dt_latest.naive_utc()),
            "raise" => polars_bail!(ComputeError:
                format!("datetime '{}' is ambiguous in time zone '{}'. \
                    Please use `ambiguous` to tell how it should be localized. \
                    If you got here from a function which doesn't have a `ambiguous` argument, \
                    please open an issue at https://github.com/pola-rs/polars/issues.", ndt, tz)
            ),
            ambiguous => polars_bail!(ComputeError:
                format!("Invalid argument {}, expected one of: \"earliest\", \"latest\", \"raise\"", ambiguous)
            ),
        },
        LocalResult::None => {
            polars_bail!(
                ComputeError: format!("datetime '{}' is non-existent in time zone '{}'. Non-existent datetimes are not yet supported", ndt, tz)
            )
        },
    }
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_datetime(ndt: NaiveDateTime, tz: &Tz) -> NaiveDateTime {
    // e.g. '2021-01-01 03:00CDT' -> '2021-01-01 03:00'
    tz.from_utc_datetime(&ndt).naive_local()
}

#[cfg(feature = "timezones")]
pub(crate) fn localize_timestamp(timestamp: i64, tu: TimeUnit, tz: Tz) -> PolarsResult<i64> {
    let dt = timestamp_to_naive_datetime_method(&tu)(timestamp);
    let localized = localize_datetime(dt, &tz, "raise")?;

    match tu {
        TimeUnit::Nanoseconds => Ok(localized.timestamp_nanos_opt().unwrap()),
        TimeUnit::Microseconds => Ok(localized.timestamp_micros()),
        TimeUnit::Milliseconds => Ok(localized.timestamp_millis()),
    }
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_timestamp(timestamp: i64, tu: TimeUnit, tz: Tz) -> i64 {
    let ts = timestamp_to_naive_datetime_method(&tu)(timestamp);
    let unlocalized = unlocalize_datetime(ts, &tz);

    match tu {
        TimeUnit::Nanoseconds => unlocalized.timestamp_nanos_opt().unwrap(),
        TimeUnit::Microseconds => unlocalized.timestamp_micros(),
        TimeUnit::Milliseconds => unlocalized.timestamp_millis(),
    }
}
