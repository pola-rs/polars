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
mod utf8;

pub use self::conversion::*;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

pub fn unix_time() -> NaiveDateTime {
    NaiveDateTime::from_timestamp(0, 0)
}
