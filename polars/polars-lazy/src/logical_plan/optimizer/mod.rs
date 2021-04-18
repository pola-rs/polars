use crate::prelude::*;
use ahash::RandomState;
use polars_core::prelude::*;
use std::collections::HashMap;

pub(crate) mod aggregate_pushdown;
pub(crate) mod aggregate_scan_projections;
pub(crate) mod predicate_pushdown;
pub(crate) mod projection_pushdown;
pub(crate) mod simplify_expr;
pub(crate) mod stack_opt;
pub(crate) mod type_coercion;

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan>;
}

// arbitrary constant to reduce reallocation.
const HASHMAP_SIZE: usize = 32;

pub(crate) fn init_hashmap<K, V>() -> HashMap<K, V, RandomState> {
    HashMap::with_capacity_and_hasher(HASHMAP_SIZE, RandomState::new())
}
