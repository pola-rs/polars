use crate::prelude::*;
use polars_core::{datatypes::PlHashMap, prelude::*};

pub(crate) mod aggregate_pushdown;
#[cfg(any(feature = "parquet", feature = "csv-file"))]
pub(crate) mod aggregate_scan_projections;
pub(crate) mod delay_rechunk;
pub(crate) mod drop_nulls;
pub(crate) mod fast_projection;
pub(crate) mod predicate_pushdown;
pub(crate) mod projection_pushdown;
pub(crate) mod simplify_expr;
mod slice_pushdown_expr;
pub mod slice_pushdown_lp;
pub(crate) mod stack_opt;
pub(crate) mod type_coercion;

use crate::prelude::stack_opt::OptimizationRule;

pub(crate) use slice_pushdown_lp::SlicePushDown;

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan>;
}

// arbitrary constant to reduce reallocation.
const HASHMAP_SIZE: usize = 16;

pub(crate) fn init_hashmap<K, V>() -> PlHashMap<K, V> {
    PlHashMap::with_capacity(HASHMAP_SIZE)
}
