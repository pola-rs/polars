use crate::lazy::prelude::*;
use crate::lazy::utils::expr_to_root_column_expr;
use crate::prelude::*;
use ahash::RandomState;
use std::collections::HashMap;

pub(crate) mod aggregate_scan_projections;
pub(crate) mod predicate;
pub(crate) mod projection;
pub(crate) mod simplify_expr;
pub(crate) mod type_coercion;

// check if a selection/projection can be done on the downwards schema
fn check_down_node(expr: &Expr, down_schema: &Schema) -> bool {
    match expr_to_root_column_expr(expr) {
        Err(_) => false,
        Ok(e) => e.to_field(down_schema).is_ok(),
    }
}

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan>;
}

// arbitrary constant to reduce reallocation.
// don't expect more than 100 predicates.
const HASHMAP_SIZE: usize = 100;

fn init_hashmap<K, V>() -> HashMap<K, V, RandomState> {
    HashMap::with_capacity_and_hasher(HASHMAP_SIZE, RandomState::new())
}
