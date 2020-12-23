use crate::lazy::logical_plan::ALogicalPlanBuilder;
use crate::lazy::prelude::*;
use crate::lazy::utils::has_aexpr;

pub(crate) struct AggregatePushdown {}

impl OptimizationRule for AggregatePushdown {
    fn optimize_plan(
        &self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.get_mut(node);
        use ALogicalPlan::*;
        match lp {
            Projection {
                expr,
                input,
                schema,
            } => {
                for node in expr {
                    // has_aexpr(node, expr_arena, AExpr::)
                }
                todo!()
            }
            _ => None,
        }
    }
}
