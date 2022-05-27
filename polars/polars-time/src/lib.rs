#![cfg_attr(docsrs, feature(doc_cfg))]
pub mod chunkedarray;
mod date_range;
mod groupby;
pub mod prelude;
pub mod series;
mod truncate;
mod upsample;
mod windows;

#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
pub use groupby::dynamic::*;

pub use {
    date_range::*, truncate::*, upsample::*, windows::calendar::date_range as date_range_vec,
    windows::duration::Duration, windows::groupby::ClosedWindow, windows::window::Window,
};
