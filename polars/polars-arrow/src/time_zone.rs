use std::fmt::{Debug, Display};
#[cfg(not(feature = "timezones"))]
pub trait PolarsTimeZone: Debug + Display + Sync + Send + Clone {}

// a dummy type that implements required subtraits
#[cfg(not(feature = "timezones"))]
impl PolarsTimeZone for u8 {}
#[cfg(not(feature = "timezones"))]
pub const NO_TIMEZONE: Option<&u8> = None;

#[cfg(feature = "timezones")]
use chrono::FixedOffset;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
#[cfg(feature = "timezones")]
pub trait PolarsTimeZone: chrono::TimeZone + Debug + Display + Sync + Send + Clone {}
#[cfg(feature = "timezones")]
impl PolarsTimeZone for FixedOffset {}
#[cfg(feature = "timezones")]
impl PolarsTimeZone for Tz {}

#[cfg(feature = "timezones")]
pub const NO_TIMEZONE: Option<&FixedOffset> = None;
