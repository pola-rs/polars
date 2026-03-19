mod date_like;
mod group_by;
mod joins;
mod list;
mod ops;
#[cfg(feature = "rolling_window")]
mod rolling_window;
mod series;
mod utils;

use polars::prelude::*;
