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
#[cfg(feature = "dtype-date")]
use chrono::NaiveDate;
use chrono::NaiveDateTime;
#[cfg(any(feature = "dtype-time", feature = "dtype-date"))]
use chrono::NaiveTime;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "dtype-time")]
pub use time::time_to_time64ns;

pub use self::conversion::*;
#[cfg(feature = "timezones")]
use crate::prelude::{polars_bail, PolarsResult};

pub fn unix_time() -> NaiveDateTime {
    NaiveDateTime::from_timestamp_opt(0, 0).unwrap()
}

#[cfg(feature = "timezones")]
pub(crate) fn validate_time_zone(tz: &str) -> PolarsResult<()> {
    match tz.parse::<Tz>() {
        Ok(_) => Ok(()),
        Err(_) => {
            polars_bail!(ComputeError: "unable to parse time zone: '{}'. Please check the Time Zone Database for a list of available time zones", tz)
        },
    }
}

#[cfg(feature = "timezones")]
pub fn parse_time_zone(tz: &str) -> PolarsResult<Tz> {
    match tz.parse::<Tz>() {
        Ok(tz) => Ok(tz),
        Err(_) => {
            polars_bail!(ComputeError: "unable to parse time zone: '{}'. Please check the Time Zone Database for a list of available time zones", tz)
        },
    }
}
