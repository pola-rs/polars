#![cfg_attr(docsrs, feature(doc_auto_cfg))]
pub mod chunkedarray;
mod date_range;
mod groupby;
pub mod prelude;
mod round;
pub mod series;
mod truncate;
mod upsample;
mod utils;
mod windows;

use chrono::FixedOffset;
pub use date_range::*;
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
pub use groupby::dynamic::*;
pub use round::*;
pub use truncate::*;
pub use upsample::*;
pub use windows::calendar::date_range as date_range_vec;
pub use windows::duration::Duration;
pub use windows::groupby::ClosedWindow;
pub use windows::window::Window;
pub const NO_TIMEZONE: Option<&FixedOffset> = None;
#[cfg(feature = "timezones")]
use chrono_tz::Tz;
pub trait PolarsTimeZone: chrono::TimeZone + std::fmt::Debug + std::fmt::Display {}
impl PolarsTimeZone for FixedOffset {}
#[cfg(feature = "timezones")]
impl PolarsTimeZone for Tz {}
