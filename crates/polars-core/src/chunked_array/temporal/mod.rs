//! Traits and utilities for temporal data.
pub mod conversion;
#[cfg(feature = "dtype-date")]
mod date;
#[cfg(feature = "dtype-datetime")]
mod datetime;
#[cfg(feature = "dtype-duration")]
mod duration;
#[cfg(feature = "dtype-time")]
mod time;

#[cfg(feature = "timezones")]
use arrow::legacy::kernels::{Ambiguous, NonExistent, convert_to_naive_local};
#[cfg(feature = "dtype-date")]
use chrono::NaiveDate;
use chrono::NaiveDateTime;
#[cfg(any(feature = "dtype-time", feature = "dtype-date"))]
use chrono::NaiveTime;
#[cfg(feature = "timezones")]
use chrono::TimeZone as _;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
use polars_error::PolarsResult;
#[cfg(feature = "timezones")]
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "dtype-time")]
pub use time::time_to_time64ns;

pub use self::conversion::*;

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
pub fn try_localize_datetime(
    ndt: NaiveDateTime,
    tz: &Tz,
    ambiguous: Ambiguous,
    non_existent: NonExistent,
) -> PolarsResult<Option<NaiveDateTime>> {
    convert_to_naive_local(&chrono_tz::UTC, tz, ndt, ambiguous, non_existent)
}

#[cfg(feature = "timezones")]
pub fn unlocalize_datetime(ndt: NaiveDateTime, tz: &Tz) -> NaiveDateTime {
    // e.g. '2021-01-01 03:00CDT' -> '2021-01-01 03:00'
    tz.from_utc_datetime(&ndt).naive_local()
}
