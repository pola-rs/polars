use crate::lazy::prelude::*;
use crate::lazy::utils::expr_to_root_column_expr;
use crate::prelude::*;

pub(crate) mod predicate;
pub(crate) mod projection;
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
