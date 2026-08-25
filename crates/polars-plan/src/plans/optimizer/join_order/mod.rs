//! Join reordering.
//!
//! Without this, joins run in the order they were written, so a large fact table
//! listed before a filtered dimension is carried at full width through every join.
//!
//! This pass finds runs of inner equi-joins, estimates how many rows each ordering
//! produces, and rebuilds the run smallest-first. The estimates can be wrong, so it
//! skips any cluster it cannot measure and projects the result back to the original
//! schema.
//!
//! - [`cluster`] decides what may be reordered,
//! - [`stats`] estimates sizes,
//! - [`enumerate`] picks the order,
//! - [`rebuild`] emits the new plan.

use polars_core::error::PolarsResult;
use polars_core::prelude::PlIndexMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use recursive::recursive;

use crate::plans::{AExpr, IR};

mod cluster;
mod enumerate;
mod rebuild;
mod stats;

/// Reorder joins throughout the plan, returning the new root.
pub(super) fn join_order(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Node> {
    // Common-subplan elimination runs earlier, so a cached subtree can be reached
    // from several parents. Rewriting it once per parent would give caches that
    // share an id but not a plan.
    let mut rewritten = PlIndexMap::default();
    rewrite(root, ir_arena, expr_arena, &mut rewritten)
}

#[recursive]
fn rewrite(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    rewritten: &mut PlIndexMap<Node, Node>,
) -> PolarsResult<Node> {
    if let Some(&done) = rewritten.get(&node) {
        return Ok(done);
    }
    // Top-down, so the outermost join of a run forms the cluster and the whole run
    // is considered together.
    if let Some(mut cluster) = cluster::extract(node, ir_arena, expr_arena) {
        // Leaves are opaque to reordering but may contain clusters of their own.
        for leaf in &mut cluster.leaves {
            leaf.node = rewrite(leaf.node, ir_arena, expr_arena, rewritten)?;
        }

        let order = enumerate::order(&cluster);
        let new_node = rebuild::rebuild(&cluster, &order, ir_arena, expr_arena)?;
        rewritten.insert(node, new_node);
        return Ok(new_node);
    }

    // `UnitVec` keeps the single-input case off the heap.
    let children = ir_arena.get(node).get_inputs();
    let mut new_children = UnitVec::with_capacity(children.len());
    for child in children {
        new_children.push(rewrite(child, ir_arena, expr_arena, rewritten)?);
    }

    for (slot, new) in ir_arena.get_mut(node).inputs_mut().zip(new_children) {
        *slot = new;
    }

    rewritten.insert(node, node);
    Ok(node)
}
