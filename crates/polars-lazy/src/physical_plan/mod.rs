pub mod executors;
#[cfg(any(feature = "list_eval", feature = "pivot"))]
pub(crate) mod exotic;
pub mod planner;
#[cfg(feature = "streaming")]
pub(crate) mod streaming;

use polars_core::prelude::*;

use crate::prelude::*;
