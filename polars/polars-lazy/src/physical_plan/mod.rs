pub mod executors;
pub mod expressions;
pub mod planner;
pub(crate) mod state;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::prelude::*;
use polars_io::PhysicalIoExpr;

/// A type that implements this transforms a LogicalPlan to a physical plan.
///
/// We could produce different physical plans with different goals in mind, e.g. memory optimized
/// performance optimized, out of core, etc.
pub trait PhysicalPlanner {
    fn create_physical_plan(
        &self,
        root: Node,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<Box<dyn Executor>>;
}

// Executor are the executors of the physical plan and produce DataFrames. They
// combine physical expressions, which produce Series.

/// Executors will evaluate physical expressions and collect them in a DataFrame.
///
/// Executors have other executors as input. By having a tree of executors we can execute the
/// physical plan until the last executor is evaluated.
pub trait Executor: Send + Sync {
    fn execute(&mut self, cache: &ExecutionState) -> Result<DataFrame>;
}
