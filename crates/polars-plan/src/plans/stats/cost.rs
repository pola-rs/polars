//! A rough cost for evaluating a subplan.

use polars_utils::aliases::InitHashMaps;
#[expect(clippy::disallowed_types)] // We don't iterate over it.
use polars_utils::aliases::PlHashSet;
use polars_utils::arena::{Arena, Node};

use super::node_stats;
use crate::plans::{AExpr, IR};

/// What a subplan costs and what it produces.
pub struct SubplanCost {
    /// Work in evaluating it, as the rows every node in it emits summed together.
    pub work: f64,
    /// Rows the subplan itself emits.
    pub rows: f64,
}

/// Estimate what the subplan rooted at `node` costs.
///
/// `None` when any node of it is not modelled: a plan we cannot describe must never
/// come out looking cheap.
///
/// A `Cache` below `node` is materialized once however its own references are
/// resolved, so it counts for the rows read out of it and not for the work that
/// fills it.
pub(crate) fn subplan_cost(
    node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Option<SubplanCost> {
    let rows = node_stats(node, ir_arena, expr_arena)?.filtered;
    let mut work = 0.0;
    let mut stack = vec![node];
    let mut inputs = Vec::new();
    #[expect(clippy::disallowed_types)] // We don't iterate over it.
    let mut seen = PlHashSet::new();

    while let Some(current) = stack.pop() {
        if !seen.insert(current) {
            continue;
        }
        work += if current == node {
            rows
        } else {
            node_stats(current, ir_arena, expr_arena)?.filtered
        };

        if current != node && matches!(ir_arena.get(current), IR::Cache { .. }) {
            continue;
        }

        inputs.clear();
        ir_arena.get(current).copy_inputs(&mut inputs);
        stack.extend(inputs.iter().copied());
    }
    Some(SubplanCost { work, rows })
}
