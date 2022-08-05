#[cfg(feature = "log")]
mod log;
#[cfg(feature = "rolling_window")]
mod rolling;
mod various;

#[cfg(feature = "log")]
pub use log::*;
use polars_core::prelude::*;

#[cfg(feature = "rolling_window")]
pub use rolling::*;
pub use various::*;

pub trait SeriesSealed {
    fn as_series(&self) -> &Series;
}

impl SeriesSealed for Series {
    fn as_series(&self) -> &Series {
        self
    }
}
