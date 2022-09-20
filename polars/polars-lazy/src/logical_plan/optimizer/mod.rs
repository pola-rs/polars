use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;

use crate::prelude::*;

pub(crate) mod aggregate_pushdown;
pub(crate) mod cache_states;
#[cfg(feature = "cse")]
pub(crate) mod cse;
pub(crate) mod delay_rechunk;
pub(crate) mod drop_nulls;
pub(crate) mod fast_projection;
#[cfg(any(
    feature = "ipc",
    feature = "parquet",
    feature = "csv-file",
    feature = "cse"
))]
pub(crate) mod file_caching;
pub(crate) mod predicate_pushdown;
pub(crate) mod projection_pushdown;
pub(crate) mod simplify_expr;
mod slice_pushdown_expr;
pub mod slice_pushdown_lp;
pub(crate) mod stack_opt;
pub(crate) mod type_coercion;

pub(crate) use slice_pushdown_lp::SlicePushDown;

use crate::prelude::stack_opt::OptimizationRule;

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> PolarsResult<LogicalPlan>;
}

// arbitrary constant to reduce reallocation.
const HASHMAP_SIZE: usize = 16;

pub(crate) fn init_hashmap<K, V>(max_len: Option<usize>) -> PlHashMap<K, V> {
    PlHashMap::with_capacity(std::cmp::min(max_len.unwrap_or(HASHMAP_SIZE), HASHMAP_SIZE))
}
