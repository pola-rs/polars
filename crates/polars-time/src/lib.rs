#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature
// combinations
#[cfg(feature = "timezones")]
mod base_utc_offset;
pub mod chunkedarray;
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
mod date_range;
#[cfg(feature = "timezones")]
mod dst_offset;
mod group_by;
#[cfg(feature = "month_end")]
mod month_end;
#[cfg(feature = "month_start")]
mod month_start;
#[cfg(feature = "offset_by")]
mod offset_by;
pub mod prelude;
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
pub mod replace;
mod round;
pub mod series;
mod truncate;
mod upsample;
mod utils;
mod windows;

#[cfg(feature = "timezones")]
pub use base_utc_offset::*;
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
pub use date_range::*;
#[cfg(feature = "timezones")]
pub use dst_offset::*;
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
pub use group_by::dynamic::*;
#[cfg(feature = "month_end")]
pub use month_end::*;
#[cfg(feature = "month_start")]
pub use month_start::*;
#[cfg(feature = "offset_by")]
pub use offset_by::*;
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
pub use replace::*;
pub use round::*;
#[cfg(feature = "dtype-date")]
pub use truncate::*;
pub use upsample::*;
#[cfg(feature = "timezones")]
pub use utils::known_timezones;
pub use windows::duration::Duration;
pub use windows::group_by::ClosedWindow;
pub use windows::window::Window;
