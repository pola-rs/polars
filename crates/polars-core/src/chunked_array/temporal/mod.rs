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
use jiff::civil::Date as NaiveDate;
use jiff::civil::DateTime as NaiveDateTime;
#[cfg(any(feature = "dtype-time", feature = "dtype-date"))]
use jiff::civil::Time as NaiveTime;
#[cfg(feature = "timezones")]
use jiff::tz::TimeZone as Tz;
#[cfg(feature = "timezones")]
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "dtype-time")]
pub use time::time_to_time64ns;

pub use self::conversion::*;
