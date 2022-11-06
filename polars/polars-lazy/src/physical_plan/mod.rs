mod errors;
pub mod executors;
#[cfg(any(feature = "list_eval", feature = "pivot"))]
pub(crate) mod exotic;
pub mod expressions;
#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
mod file_cache;
mod node_timer;
pub mod planner;
pub(crate) mod state;
#[cfg(feature = "streaming")]
pub(crate) mod streaming;

use errors::expression_err;
use polars_core::prelude::*;
use polars_io::predicates::PhysicalIoExpr;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
