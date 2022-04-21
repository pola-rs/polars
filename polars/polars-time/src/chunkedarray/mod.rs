//! Traits and utilities for temporal data.
#[cfg(feature = "dtype-date")]
mod date;
#[cfg(feature = "dtype-datetime")]
mod datetime;
#[cfg(feature = "dtype-duration")]
mod duration;
mod kernels;
#[cfg(feature = "dtype-time")]
mod time;
pub mod utf8;

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
#[cfg(feature = "dtype-date")]
pub use date::DateMethods;
#[cfg(feature = "dtype-datetime")]
pub use datetime::DatetimeMethods;
#[cfg(feature = "dtype-duration")]
pub use duration::DurationMethods;
use kernels::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::prelude::*;
#[cfg(feature = "dtype-time")]
pub use time::TimeMethods;
pub use utf8::Utf8Methods;

pub fn unix_time() -> NaiveDateTime {
    NaiveDateTime::from_timestamp(0, 0)
}
