//! A rough cost for evaluating a subplan.

use polars_utils::aliases::InitHashMaps;
#[expect(clippy::disallowed_types)] // We don't iterate over it.
use polars_utils::aliases::PlHashSet;
use polars_utils::arena::{Arena, Node};

use super::node::{StatsCache, node_stats_with_cache};
use crate::plans::{AExpr, IR};

/// What a subplan costs and what it produces.
pub(crate) struct SubplanCost {
    /// Work in evaluating it, as the rows every node in it emits summed together.
    pub(crate) work: f64,
    /// Rows the subplan itself emits.
    pub(crate) rows: f64,
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
    // One cache for the whole walk: every node here is an ancestor or a descendant of
    // the others, so estimating them one at a time would re-walk the same subtrees.
    let cache = &mut StatsCache::new();
    let rows = node_stats_with_cache(node, ir_arena, expr_arena, cache)?.filtered;
    let mut work = 0.0;
    let mut stack = vec![node];
    let mut inputs = Vec::new();
    #[expect(clippy::disallowed_types)] // We don't iterate over it.
    let mut seen = PlHashSet::new();

    while let Some(current) = stack.pop() {
        if !seen.insert(current) {
            continue;
        }
        work += node_stats_with_cache(current, ir_arena, expr_arena, cache)?.filtered;

        if current != node && matches!(ir_arena.get(current), IR::Cache { .. }) {
            continue;
        }

        inputs.clear();
        ir_arena.get(current).copy_inputs(&mut inputs);
        stack.extend(inputs.iter().copied());
    }
    Some(SubplanCost { work, rows })
}
