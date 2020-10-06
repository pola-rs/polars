use crate::lazy::prelude::*;
use crate::prelude::*;
pub(crate) mod predicate;
pub(crate) mod projection;

// check if a selection/projection can be done on the downwards schema
fn check_down_node(expr: &Expr, down_schema: &Schema) -> bool {
    expr.to_field(down_schema).is_ok()
}

pub trait Optimize {
    fn optimize(&self, logical_plan: LogicalPlan) -> LogicalPlan;
}
