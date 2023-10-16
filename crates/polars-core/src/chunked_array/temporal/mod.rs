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
use crate::prelude::{ArrayRef, LargeStringArray};

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

pub(crate) fn validate_is_number(vec_array: &Vec<ArrayRef>) -> bool {
    vec_array.iter().all(|array| is_parsable_as_number(array))
}

fn is_parsable_as_number(array: &ArrayRef) -> bool {
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        array.iter().all(|value| {
            value
                .expect("Unable to parse int string to datetime")
                .parse::<i64>()
                .is_ok()
        })
    } else {
        false
    }
}
