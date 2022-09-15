pub mod executors;
#[cfg(any(feature = "list_eval", feature = "pivot"))]
pub(crate) mod exotic;
pub mod expressions;
#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv-file"))]
mod file_cache;
mod node_timer;
pub mod planner;
pub(crate) mod state;

use polars_core::prelude::*;
use polars_io::predicates::PhysicalIoExpr;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

// Executor are the executors of the physical plan and produce DataFrames. They
// combine physical expressions, which produce Series.

/// Executors will evaluate physical expressions and collect them in a DataFrame.
///
/// Executors have other executors as input. By having a tree of executors we can execute the
/// physical plan until the last executor is evaluated.
pub trait Executor: Send + Sync {
    fn execute(&mut self, cache: &mut ExecutionState) -> PolarsResult<DataFrame>;
}

pub struct Dummy {}
impl Executor for Dummy {
    fn execute(&mut self, _cache: &mut ExecutionState) -> PolarsResult<DataFrame> {
        panic!("should not get here");
    }
}

impl Default for Box<dyn Executor> {
    fn default() -> Self {
        Box::new(Dummy {})
    }
}
