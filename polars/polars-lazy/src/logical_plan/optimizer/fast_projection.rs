use polars_core::prelude::*;

use crate::logical_plan::alp::ALogicalPlan;
use crate::logical_plan::functions::FunctionNode;
use crate::prelude::stack_opt::OptimizationRule;
use crate::prelude::*;

/// Projection in the physical plan is done by selecting an expression per thread.
/// In case of many projections and columns this can be expensive when the expressions are simple
/// column selections. These can be selected on a single thread. The single thread is faster, because
/// the eager selection algorithm hashes the column names, making the projection complexity linear
/// instead of quadratic.
///
/// It is important that this optimization is ran after projection pushdown.
///
/// The schema reported after this optimization is also
pub(crate) struct FastProjection {}

fn impl_fast_projection(
    input: Node,
    expr: &[Node],
    expr_arena: &mut Arena<AExpr>,
) -> Option<ALogicalPlan> {
    let mut columns = Vec::with_capacity(expr.len());
    for node in expr.iter() {
        if let AExpr::Column(name) = expr_arena.get(*node) {
            columns.push(name.clone())
        } else {
            break;
        }
    }
    if columns.len() == expr.len() {
        let lp = ALogicalPlan::MapFunction {
            input,
            function: FunctionNode::FastProjection {
                columns: Arc::new(columns),
            },
        };

        Some(lp)
    } else {
        None
    }
}

impl OptimizationRule for FastProjection {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.get(node);

        match lp {
            ALogicalPlan::Projection { input, expr, .. } => {
                if !matches!(lp_arena.get(*input), ALogicalPlan::ExtContext { .. }) {
                    impl_fast_projection(*input, expr, expr_arena)
                } else {
                    None
                }
            }
            ALogicalPlan::LocalProjection { input, expr, .. } => {
                impl_fast_projection(*input, expr, expr_arena)
            }
            _ => None,
        }
    }
}
