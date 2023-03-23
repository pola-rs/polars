#![cfg_attr(feature = "simd", feature(portable_simd))]
pub mod array;
pub mod bit_util;
mod bitmap;
pub mod compute;
pub mod conversion;
pub mod data_types;
pub mod error;
pub mod export;
pub mod floats;
pub mod index;
pub mod is_valid;
pub mod kernels;
pub mod prelude;
pub mod slice;
pub mod trusted_len;
pub mod utils;

#[cfg(not(feature = "timezones"))]
pub trait PolarsTimeZone:
    std::fmt::Debug + std::fmt::Display + std::marker::Sync + std::marker::Send
{
}

#[cfg(feature = "timezones")]
use chrono::FixedOffset;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
pub trait PolarsTimeZone:
    chrono::TimeZone + std::fmt::Debug + std::fmt::Display + std::marker::Sync + std::marker::Send
{
}
#[cfg(feature = "timezones")]
impl PolarsTimeZone for FixedOffset {}
#[cfg(feature = "timezones")]
impl PolarsTimeZone for Tz {}
