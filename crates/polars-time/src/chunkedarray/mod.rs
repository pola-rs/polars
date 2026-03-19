//! Traits and utilities for temporal data.
#[cfg(feature = "dtype-date")]
mod date;
#[cfg(feature = "dtype-datetime")]
mod datetime;
#[cfg(feature = "dtype-duration")]
mod duration;
mod kernels;
#[cfg(any(feature = "rolling_window", feature = "rolling_window_by"))]
mod rolling_window;
pub mod string;
#[cfg(feature = "dtype-time")]
mod time;

use arrow::legacy::utils::CustomIterTools;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
#[cfg(feature = "dtype-date")]
pub use date::DateMethods;
#[cfg(feature = "dtype-datetime")]
pub use datetime::DatetimeMethods;
#[cfg(feature = "dtype-duration")]
pub use duration::DurationMethods;
use kernels::*;
use polars_core::prelude::*;
#[cfg(any(feature = "rolling_window", feature = "rolling_window_by"))]
pub use rolling_window::*;
pub use string::StringMethods;
#[cfg(feature = "dtype-time")]
pub use time::TimeMethods;

// a separate function so that it is not compiled twice
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
pub(crate) fn months_to_quarters(mut ca: Int8Chunked) -> Int8Chunked {
    ca.apply_mut(|month| (month + 2) / 3);
    ca
}
