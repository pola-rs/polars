mod approx_algo;
#[cfg(feature = "approx_unique")]
mod approx_unique;
mod arg_min_max;
mod clip;
#[cfg(feature = "cum_agg")]
mod cum_agg;
#[cfg(feature = "cutqcut")]
mod cut;
#[cfg(feature = "diff")]
mod diff;
#[cfg(feature = "ewma")]
mod ewm;
#[cfg(feature = "round_series")]
mod floor_divide;
#[cfg(feature = "fused")]
mod fused;
mod horizontal;
#[cfg(feature = "convert_index")]
mod index;
#[cfg(feature = "is_first_distinct")]
mod is_first_distinct;
#[cfg(feature = "is_in")]
mod is_in;
#[cfg(feature = "is_last_distinct")]
mod is_last_distinct;
#[cfg(feature = "is_unique")]
mod is_unique;
#[cfg(feature = "log")]
mod log;
#[cfg(feature = "moment")]
mod moment;
#[cfg(feature = "pct_change")]
mod pct_change;
#[cfg(feature = "rank")]
mod rank;
#[cfg(feature = "rle")]
mod rle;
#[cfg(feature = "rolling_window")]
mod rolling;
#[cfg(feature = "round_series")]
mod round;
#[cfg(feature = "search_sorted")]
mod search_sorted;
#[cfg(feature = "to_dummies")]
mod to_dummies;
mod various;

pub use approx_algo::*;
#[cfg(feature = "approx_unique")]
pub use approx_unique::*;
pub use arg_min_max::ArgAgg;
pub use clip::*;
#[cfg(feature = "cum_agg")]
pub use cum_agg::*;
#[cfg(feature = "cutqcut")]
pub use cut::*;
#[cfg(feature = "diff")]
pub use diff::*;
#[cfg(feature = "ewma")]
pub use ewm::*;
#[cfg(feature = "round_series")]
pub use floor_divide::*;
#[cfg(feature = "fused")]
pub use fused::*;
pub use horizontal::*;
#[cfg(feature = "convert_index")]
pub use index::*;
#[cfg(feature = "is_first_distinct")]
pub use is_first_distinct::*;
#[cfg(feature = "is_in")]
pub use is_in::*;
#[cfg(feature = "is_last_distinct")]
pub use is_last_distinct::*;
#[cfg(feature = "is_unique")]
pub use is_unique::*;
#[cfg(feature = "log")]
pub use log::*;
#[cfg(feature = "moment")]
pub use moment::*;
#[cfg(feature = "pct_change")]
pub use pct_change::*;
use polars_core::prelude::*;
#[cfg(feature = "rank")]
pub use rank::*;
#[cfg(feature = "rle")]
pub use rle::*;
#[cfg(feature = "rolling_window")]
pub use rolling::*;
#[cfg(feature = "round_series")]
pub use round::*;
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
