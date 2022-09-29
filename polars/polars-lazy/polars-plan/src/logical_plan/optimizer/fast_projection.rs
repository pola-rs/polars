use polars_core::prelude::*;

use super::*;
use crate::logical_plan::alp::ALogicalPlan;
use crate::logical_plan::functions::FunctionNode;

/// Projection in the physical plan is done by selecting an expression per thread.
/// In case of many projections and columns this can be expensive when the expressions are simple
/// column selections. These can be selected on a single thread. The single thread is faster, because
/// the eager selection algorithm hashes the column names, making the projection complexity linear
/// instead of quadratic.
///
/// It is important that this optimization is ran after projection pushdown.
///
/// The schema reported after this optimization is also
pub(super) struct FastProjectionAndCollapse {}

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

impl OptimizationRule for FastProjectionAndCollapse {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        use ALogicalPlan::*;
        let lp = lp_arena.get(node);

        match lp {
            Projection { input, expr, .. } => {
                if !matches!(lp_arena.get(*input), ExtContext { .. }) {
                    impl_fast_projection(*input, expr, expr_arena)
                } else {
                    None
                }
            }
            LocalProjection { input, expr, .. } => impl_fast_projection(*input, expr, expr_arena),
            MapFunction {
                input,
                function: FunctionNode::FastProjection { columns },
            } => {
                // if there are 2 subsequent fast projections, flatten them and only take the last
                match lp_arena.get(*input) {
                    MapFunction {
                        function: FunctionNode::FastProjection { .. },
                        input: prev_input,
                    } => Some(MapFunction {
                        input: *prev_input,
                        function: FunctionNode::FastProjection {
                            columns: columns.clone(),
                        },
                    }),
                    // cleanup projections set in projection pushdown just above caches
                    // they are nto needed.
                    cache_lp @ Cache { .. } => {
                        if cache_lp.schema(lp_arena).len() == columns.len() {
                            Some(cache_lp.clone())
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            // if there are 2 subsequent caches, flatten them and only take the inner
            Cache {
                input,
                count: outer_count,
                ..
            } => {
                if let Cache {
                    input: prev_input,
                    id,
                    count,
                } = lp_arena.get(*input)
                {
                    Some(Cache {
                        input: *prev_input,
                        id: *id,
                        // ensure the counts are updated
                        count: count.saturating_add(*outer_count),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
