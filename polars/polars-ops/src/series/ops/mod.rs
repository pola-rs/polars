#[cfg(feature = "log")]
mod log;
#[cfg(feature = "log")]
pub use log::*;
use polars_core::prelude::*;

pub trait SeriesSealed {
    fn as_series(&self) -> &Series;
}

impl SeriesSealed for Series {
    fn as_series(&self) -> &Series {
        self
    }
}
