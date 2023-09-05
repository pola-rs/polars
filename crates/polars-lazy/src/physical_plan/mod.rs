pub mod executors;
#[cfg(any(feature = "list_eval", feature = "pivot"))]
pub(crate) mod exotic;
pub mod expressions;
#[cfg(any(
    feature = "ipc",
    feature = "parquet",
    feature = "csv",
    feature = "json"
))]
mod file_cache;
mod node_timer;
pub mod planner;
pub(crate) mod state;
#[cfg(feature = "streaming")]
pub(crate) mod streaming;

use polars_core::prelude::*;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
