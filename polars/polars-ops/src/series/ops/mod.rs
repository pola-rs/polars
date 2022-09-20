#[cfg(feature = "log")]
mod log;
#[cfg(feature = "rolling_window")]
mod rolling;
#[cfg(feature = "search_sorted")]
mod search_sorted;
mod various;

#[cfg(feature = "log")]
pub use log::*;
use polars_core::prelude::*;
#[cfg(feature = "rolling_window")]
pub use rolling::*;
#[cfg(feature = "search_sorted")]
pub use search_sorted::*;
pub use various::*;

pub trait SeriesSealed {
    fn as_series(&self) -> &Series;
}

impl SeriesSealed for Series {
    fn as_series(&self) -> &Series {
        self
    }
}
