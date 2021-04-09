pub mod executors;
pub mod expressions;
pub mod planner;

use crate::prelude::*;
use ahash::RandomState;
use polars_core::prelude::*;
use polars_io::PhysicalIoExpr;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub enum ExprVal {
    Series(Series),
    Column(Vec<String>),
}

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
pub trait Executor: Send + Sync {
    fn execute(&mut self, cache: &Cache) -> Result<DataFrame>;
}

pub(crate) type Cache = Arc<Mutex<HashMap<String, DataFrame, RandomState>>>;
