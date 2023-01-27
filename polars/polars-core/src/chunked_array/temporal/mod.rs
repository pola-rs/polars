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

pub use self::conversion::*;

pub fn unix_time() -> NaiveDateTime {
    NaiveDateTime::from_timestamp_opt(0, 0).unwrap()
}
