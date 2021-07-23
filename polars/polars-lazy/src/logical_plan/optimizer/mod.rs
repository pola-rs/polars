use crate::prelude::*;
use polars_core::{datatypes::PlHashMap, prelude::*};

pub(crate) mod aggregate_pushdown;
#[cfg(any(feature = "parquet", feature = "csv-file"))]
pub(crate) mod aggregate_scan_projections;
pub(crate) mod fast_projection;
#[cfg(feature = "private")]
pub(crate) mod join_pruning;
pub(crate) mod predicate_pushdown;
pub(crate) mod projection_pushdown;
pub(crate) mod simplify_expr;
pub(crate) mod stack_opt;
pub(crate) mod type_coercion;

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan>;
}

// arbitrary constant to reduce reallocation.
const HASHMAP_SIZE: usize = 16;

pub(crate) fn init_hashmap<K, V>() -> PlHashMap<K, V> {
    PlHashMap::with_capacity(HASHMAP_SIZE)
}
