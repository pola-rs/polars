mod expr;
mod lp;

use crate::prelude::*;
pub use expr::*;
pub use lp::*;
use polars_core::prelude::*;

impl PhysicalPlanner for DefaultPlanner {
    fn create_physical_plan(
        &self,
        _root: Node,
        _lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
    ) -> Result<Box<dyn Executor>> {
        self.create_physical_plan(_root, _lp_arena, _expr_arena)
    }
}
