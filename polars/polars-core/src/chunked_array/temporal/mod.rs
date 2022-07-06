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

pub use self::conversion::*;

use chrono::NaiveDateTime;
#[cfg(any(feature = "dtype-time", feature = "dtype-date"))]
use chrono::NaiveTime;

#[cfg(feature = "dtype-date")]
use chrono::NaiveDate;

pub fn unix_time() -> NaiveDateTime {
    NaiveDateTime::from_timestamp(0, 0)
}
