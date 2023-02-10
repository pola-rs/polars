mod arg_min_max;
#[cfg(feature = "round_series")]
mod floor_divide;
#[cfg(feature = "is_first")]
mod is_first;
#[cfg(feature = "log")]
mod log;
#[cfg(feature = "rolling_window")]
mod rolling;
#[cfg(feature = "search_sorted")]
mod search_sorted;
#[cfg(feature = "to_dummies")]
mod to_dummies;
mod various;

pub use arg_min_max::ArgAgg;
#[cfg(feature = "round_series")]
pub use floor_divide::*;
#[cfg(feature = "is_first")]
pub use is_first::*;
#[cfg(feature = "log")]
pub use log::*;
use polars_core::prelude::*;
#[cfg(feature = "rolling_window")]
pub use rolling::*;
#[cfg(feature = "search_sorted")]
pub use search_sorted::*;
#[cfg(feature = "to_dummies")]
pub use to_dummies::*;
pub use various::*;

pub trait SeriesSealed {
    fn as_series(&self) -> &Series;
}

impl SeriesSealed for Series {
    fn as_series(&self) -> &Series {
        self
    }
}
