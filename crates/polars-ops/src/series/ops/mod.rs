#[cfg(feature = "abs")]
mod abs;
#[cfg(feature = "approx_unique")]
mod approx_algo;
#[cfg(feature = "approx_unique")]
mod approx_unique;
mod arg_min_max;
#[cfg(feature = "business")]
mod business;
mod clip;
#[cfg(feature = "cum_agg")]
mod cum_agg;
#[cfg(feature = "cutqcut")]
mod cut;
#[cfg(feature = "diff")]
mod diff;
#[cfg(feature = "ewma")]
mod ewm;
#[cfg(feature = "ewma_by")]
mod ewm_by;
#[cfg(feature = "round_series")]
mod floor_divide;
#[cfg(feature = "fused")]
mod fused;
mod horizontal;
mod index;
mod int_range;
#[cfg(any(feature = "interpolate_by", feature = "interpolate"))]
mod interpolation;
#[cfg(feature = "is_between")]
mod is_between;
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
mod negate;
#[cfg(feature = "pct_change")]
mod pct_change;
#[cfg(feature = "rank")]
mod rank;
#[cfg(feature = "reinterpret")]
mod reinterpret;
#[cfg(feature = "replace")]
mod replace;
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
#[cfg(feature = "unique_counts")]
mod unique;
mod various;

#[cfg(feature = "abs")]
pub use abs::*;
#[cfg(feature = "approx_unique")]
pub use approx_algo::*;
#[cfg(feature = "approx_unique")]
pub use approx_unique::*;
pub use arg_min_max::ArgAgg;
#[cfg(feature = "business")]
pub use business::*;
pub use clip::*;
#[cfg(feature = "cum_agg")]
pub use cum_agg::*;
#[cfg(feature = "cutqcut")]
pub use cut::*;
#[cfg(feature = "diff")]
pub use diff::*;
#[cfg(feature = "ewma")]
pub use ewm::*;
#[cfg(feature = "ewma_by")]
pub use ewm_by::*;
#[cfg(feature = "round_series")]
pub use floor_divide::*;
#[cfg(feature = "fused")]
pub use fused::*;
pub use horizontal::*;
pub use index::*;
pub use int_range::*;
#[cfg(feature = "interpolate")]
pub use interpolation::interpolate::*;
#[cfg(feature = "interpolate_by")]
pub use interpolation::interpolate_by::*;
#[cfg(any(feature = "interpolate", feature = "interpolate_by"))]
pub use interpolation::*;
#[cfg(feature = "is_between")]
pub use is_between::*;
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
pub use negate::*;
#[cfg(feature = "pct_change")]
pub use pct_change::*;
pub use polars_core::chunked_array::ops::search_sorted::SearchSortedSide;
use polars_core::prelude::*;
#[cfg(feature = "rank")]
pub use rank::*;
#[cfg(feature = "reinterpret")]
pub use reinterpret::*;
#[cfg(feature = "replace")]
pub use replace::*;
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
#[cfg(feature = "unique_counts")]
pub use unique::*;
pub use various::*;
mod not;

#[cfg(feature = "dtype-duration")]
pub(crate) mod duration;
#[cfg(feature = "dtype-duration")]
pub use duration::*;
pub use not::*;

pub trait SeriesSealed {
    fn as_series(&self) -> &Series;
}

impl SeriesSealed for Series {
    fn as_series(&self) -> &Series {
        self
    }
}
