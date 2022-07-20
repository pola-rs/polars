#[cfg(feature = "log")]
mod log;
mod various;

#[cfg(feature = "log")]
pub use log::*;
use polars_core::prelude::*;
pub use various::*;

pub trait SeriesSealed {
    fn as_series(&self) -> &Series;
}

impl SeriesSealed for Series {
    fn as_series(&self) -> &Series {
        self
    }
}
