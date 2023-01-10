#![cfg_attr(docsrs, feature(doc_auto_cfg))]
pub mod chunkedarray;
mod date_range;
mod groupby;
pub mod prelude;
mod round;
pub mod series;
mod truncate;
mod upsample;
mod windows;

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
