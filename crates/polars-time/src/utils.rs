#[cfg(feature = "timezones")]
use arrow::legacy::kernels::{Ambiguous, NonExistent, convert_to_naive_local};
#[cfg(feature = "timezones")]
use arrow::legacy::time_zone::Tz;
#[cfg(feature = "timezones")]
use jiff::civil::DateTime as NaiveDateTime;
#[cfg(feature = "timezones")]
use polars_core::prelude::PolarsResult;

/// Localize datetime according to given time zone.
///
/// e.g. '2021-01-01 03:00' -> '2021-01-01 03:00CDT'
///
/// Note: this may only return `Ok(None)` if ambiguous is Ambiguous::Null
/// or if non_existent is NonExistent::Null.
/// Otherwise, it will either return `Ok(Some(NaiveDateTime))` or `PolarsError`.
///
/// Therefore, calling `try_localize_datetime(..., Ambiguous::Raise, NonExistent::Raise)?.unwrap()`
/// is safe, and will never panic.
#[cfg(feature = "timezones")]
pub(crate) fn try_localize_datetime(
    ndt: NaiveDateTime,
    tz: &Tz,
    ambiguous: Ambiguous,
    non_existent: NonExistent,
) -> PolarsResult<Option<NaiveDateTime>> {
    convert_to_naive_local(&Tz::UTC, tz, ndt, ambiguous, non_existent)
}

#[cfg(feature = "timezones")]
pub(crate) fn unlocalize_datetime(ndt: NaiveDateTime, tz: &Tz) -> NaiveDateTime {
    // e.g. '2021-01-01 03:00CDT' -> '2021-01-01 03:00'
    let ts = Tz::UTC.to_timestamp(ndt).expect("datetime out-of-range");
    tz.to_datetime(ts)
}

#[cfg(feature = "timezones")]
pub fn known_timezones() -> Vec<String> {
    jiff::tz::db()
        .available()
        .map(|name| name.as_str().to_string())
        .collect()
}
