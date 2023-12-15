#[cfg(feature = "timezones")]
use arrow::legacy::kernels::{convert_to_naive_local, convert_to_naive_local_opt, Ambiguous};
#[cfg(feature = "timezones")]
use arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use chrono::NaiveDateTime;
#[cfg(feature = "timezones")]
use chrono::TimeZone;
#[cfg(feature = "timezones")]
use polars_core::prelude::PolarsResult;

#[cfg(feature = "timezones")]
pub(crate) fn try_localize_datetime(
    ndt: NaiveDateTime,
    tz: &Tz,
    ambiguous: Ambiguous,
) -> PolarsResult<NaiveDateTime> {
    // e.g. '2021-01-01 03:00' -> '2021-01-01 03:00CDT'
    convert_to_naive_local(&chrono_tz::UTC, tz, ndt, ambiguous)
}

#[cfg(feature = "timezones")]
pub(crate) fn localize_datetime_opt(
    ndt: NaiveDateTime,
    tz: &Tz,
    ambiguous: Ambiguous,
) -> Option<NaiveDateTime> {
    // e.g. '2021-01-01 03:00' -> '2021-01-01 03:00CDT'
    convert_to_naive_local_opt(&chrono_tz::UTC, tz, ndt, ambiguous)
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_datetime(ndt: NaiveDateTime, tz: &Tz) -> NaiveDateTime {
    // e.g. '2021-01-01 03:00CDT' -> '2021-01-01 03:00'
    tz.from_utc_datetime(&ndt).naive_local()
}
