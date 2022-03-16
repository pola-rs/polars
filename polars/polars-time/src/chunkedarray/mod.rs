//! Traits and utilities for temporal data.
#[cfg(feature = "dtype-date")]
mod date;
#[cfg(feature = "dtype-datetime")]
mod datetime;
#[cfg(feature = "dtype-duration")]
mod duration;
#[cfg(feature = "dtype-time")]
mod time;
mod utf8;
mod kernels;

use std::ops::Deref;
use polars_core::prelude::*;
use kernels::*;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use polars_arrow::trusted_len::TrustedLen;
use polars_arrow::utils::CustomIterTools;
pub use date::DateMethods;
pub use datetime::DatetimeMethods;
pub use duration::DurationMethods;
pub use time::TimeMethods;
pub use utf8::Utf8Methods;

pub fn unix_time() -> NaiveDateTime {
    NaiveDateTime::from_timestamp(0, 0)
}

pub struct TemporalMethods<T>(T);

impl<T> Deref for TemporalMethods<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}