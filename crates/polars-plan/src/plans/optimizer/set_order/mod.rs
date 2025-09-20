//! Pass to obtain and optimize using exhaustive row-order information.
//!
//! This pass attaches an ordering flag to all edges between IR nodes. When this flag is `true`,
//! this edge needs to be ordered.
//!
//! The pass performs two passes over the IR graph. First, it assigns and pushes ordering down from
//! the sinks to the leaves. Second, it pulls those orderings back up from the leaves to the sinks.
//! The two passes weaken order guarantees and simplify IR nodes where possible.
//!
//! When the two passes are done, we are left with a map from all nodes to the ordering status of
//! their inputs.

mod expr_pullup;
mod expr_pushdown;
mod ir_pullup;
mod ir_pushdown;

use polars_core::prelude::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unique_id::UniqueId;

use super::IR;
use crate::plans::AExpr;
use crate::plans::ir::inputs::Inputs;

/// Optimize the orderings used in the IR plan and get the relative orderings of all edges.
///
/// All roots should be `Sink` nodes and no `SinkMultiple` or `Invalid` are allowed to be part of
/// the graph.
pub fn simplify_and_fetch_orderings(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PlHashMap<Node, UnitVec<bool>> {
    let mut leaves = Vec::new();
    let mut outputs = PlHashMap::default();
    let mut cache_proxy = PlHashMap::<UniqueId, Vec<Node>>::default();

    // Get the per-node outputs and leaves
    {
        let mut stack = Vec::new();

        for root in roots {
            assert!(matches!(ir_arena.get(*root), IR::Sink { .. }));
            outputs.insert(*root, Vec::new());
            stack.extend(
                ir_arena
                    .get(*root)
                    .inputs()
                    .enumerate()
                    .map(|(root_input_idx, node)| ((*root, root_input_idx), node)),
            );
        }

        while let Some(((parent, parent_input_idx), node)) = stack.pop() {
            let ir = ir_arena.get(node);
            let node = match ir {
                IR::Cache { id, .. } => {
                    let nodes = cache_proxy.entry(*id).or_default();
                    nodes.push(node);
                    nodes[0]
                },
                _ => node,
            };

            let outputs = outputs.entry(node).or_default();
            let has_been_visisited_before = !outputs.is_empty();
            outputs.push((parent, parent_input_idx));

            if has_been_visisited_before {
                continue;
            }

            let inputs = ir.inputs();
            if matches!(inputs, Inputs::Empty) {
                leaves.push(node);
            }
            stack.extend(
                inputs
                    .enumerate()
                    .map(|(node_input_idx, input)| ((node, node_input_idx), input)),
            );
        }
    }

    // Pushdown and optimize orders from the roots to the leaves.
    let mut orders =
        ir_pushdown::pushdown_orders(roots, ir_arena, expr_arena, &mut outputs, &cache_proxy);
    // Pullup orders from the leaves to the roots.
    ir_pullup::pullup_orders(
        &leaves,
        ir_arena,
        expr_arena,
        &mut outputs,
        &mut orders,
        &cache_proxy,
    );

    // @Hack. Since not all caches might share the same node and the input of caches might have
    // been updated, we need to ensure that all caches again have the same input.
    //
    // This can be removed when all caches with the same id share the same IR node.
    for nodes in cache_proxy.into_values() {
        let updated_node = nodes[0];
        let order = orders[&updated_node].clone();
        let IR::Cache {
            input: updated_input,
            id: _,
        } = ir_arena.get(updated_node)
        else {
            unreachable!();
        };
        let updated_input = *updated_input;
        for n in &nodes[1..] {
            let IR::Cache { input, id: _ } = ir_arena.get_mut(*n) else {
                unreachable!();
            };

            orders.insert(*n, order.clone());
            *input = updated_input;
        }
    }

    orders
}
