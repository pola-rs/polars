use std::collections::HashSet;
use crate::prelude::*;
use polars_core::prelude::*;
use ahash::RandomState;

pub struct ProjectionPushDown {}

impl ProjectionPushDown {

    /// Helper method. This pushes down current node and assigns the result to this node.
    fn pushdown_and_assign(
        &self,
        input: Node,
        mut acc_projections: Vec<Expr>,
        mut names: HashSet<Arc<String>, RandomState>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<()> {
        let alp = lp_arena.take(input);
        let lp = self.push_down(alp, acc_projections, names, projections_seen, lp_arena, expr_arena)?;
        lp_arena.replace(input, lp);
        Ok(())
    }

    /// Projection pushdown optimizer
    ///
    /// # Arguments
    ///
    /// * `AlogicalPlan` - Arena based logical plan tree representing the query.
    /// * `acc_projections` - The projections we accumulate during tree traversal.
    /// * `names` - We keep track of the names to ensure we don't do duplicate projections.
    /// * `projections_seen` - Count the number of projection operations during tree traversal.
    /// * `lp_arena` - The local memory arena for the logical plan.
    /// * `expr_arena` - The local memory arena for the expressions.
    ///
    fn push_down(
        &self,
        logical_plan: ALogicalPlan,
        mut acc_projections: Vec<Expr>,
        mut names: HashSet<Arc<String>, RandomState>,
        projections_seen: usize,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<ALogicalPlan> {
        use ALogicalPlan::*;

        match logical_plan {
            Slice { input, offset, len } => {
                self.pushdown_and_assign(input, acc_projections, names, projections_seen, lp_arena, expr_arena)?;
                Ok(Slice { input, offset, len })
            }

            lp => Ok(lp)
        }
    }
}
