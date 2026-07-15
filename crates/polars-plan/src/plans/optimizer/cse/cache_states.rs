use std::ops::ControlFlow;

use polars_error::PolarsResult;
use polars_utils::aliases::{InitHashMaps, PlIndexMap, PlIndexSet};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;

use crate::dsl::Expr;
use crate::plans::deep_copy::deep_copy_ir_delete_caches;
use crate::plans::optimizer::ir_traversal::ir_graph_traversal;
use crate::plans::optimizer::ir_traversal::storage::IRTraversalStorage;
use crate::plans::visitor::AexprNode;
use crate::plans::{AExpr, ExprIR, IR, PredicatePushDown};
use crate::traversal::visitor::{FnVisitors, SubtreeVisit};
use crate::utils::aexpr_to_leaf_names;

fn get_upper_projections(
    parent: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    names_scratch: &mut Vec<PlSmallStr>,
    found_required_columns: &mut bool,
) -> bool {
    let parent = lp_arena.get(parent);

    // During projection pushdown all accumulated.
    match parent {
        IR::SimpleProjection { columns, .. } => {
            let iter = columns.iter_names_cloned();
            names_scratch.extend(iter);
            *found_required_columns = true;
            false
        },
        IR::Filter { predicate, .. } => {
            // Also add predicate, as the projection is above the filter node.
            names_scratch.extend(aexpr_to_leaf_names(predicate.node(), expr_arena));

            true
        },
        // Only filter and projection nodes are allowed, any other node we stop.
        _ => false,
    }
}

fn get_upper_predicates(
    parent: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    predicate_scratch: &mut Vec<Expr>,
) -> bool {
    let parent = lp_arena.get(parent);

    match parent {
        IR::Filter { predicate, .. } => {
            let expr = predicate.to_expr(expr_arena);
            predicate_scratch.push(expr);
            false
        },
        IR::SimpleProjection { .. } => true,
        // Only filter and projection nodes are allowed, any other node we stop.
        _ => false,
    }
}

type TwoParents = [Option<Node>; 2];

// 1. This will ensure that all equal caches communicate the amount of columns
//    they need to project.
// 2. This will ensure we apply predicate in the subtrees below the caches.
//    If the predicate above the cache is the same for all matching caches, that filter will be
//    applied as well.
//
// # Example
// Consider this tree, where `SUB-TREE` is duplicate and can be cached.
//
//
//                         Tree
//                         |
//                         |
//    |--------------------|-------------------|
//    |                                        |
//    SUB-TREE                                 SUB-TREE
//
// STEPS:
// - 1. CSE will run and will insert cache nodes
//
//                         Tree
//                         |
//                         |
//    |--------------------|-------------------|
//    |                                        |
//    | CACHE 0                                | CACHE 0
//    |                                        |
//    SUB-TREE                                 SUB-TREE
//
// - 2. predicate and projection pushdown will run and will insert optional FILTER and PROJECTION above the caches
//
//                         Tree
//                         |
//                         |
//    |--------------------|-------------------|
//    | FILTER (optional)                      | FILTER (optional)
//    | PROJ (optional)                        | PROJ (optional)
//    |                                        |
//    | CACHE 0                                | CACHE 0
//    |                                        |
//    SUB-TREE                                 SUB-TREE
//
// # Projection optimization
// The union of the projection is determined and the projection will be pushed down.
//
//                         Tree
//                         |
//                         |
//    |--------------------|-------------------|
//    | FILTER (optional)                      | FILTER (optional)
//    | CACHE 0                                | CACHE 0
//    |                                        |
//    SUB-TREE                                 SUB-TREE
//    UNION PROJ (optional)                    UNION PROJ (optional)
//
// # Filter optimization
// Depending on the predicates the predicate pushdown optimization will run.
// Possible cases:
// - NO FILTERS: run predicate pd from the cache nodes -> finish
// - Above the filters the caches are the same -> run predicate pd from the filter node -> finish
// - There is a cache without predicates above the cache node -> run predicate form the cache nodes -> finish
// - The predicates above the cache nodes are all different -> remove the cache nodes -> finish
pub(crate) fn set_cache_states(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    scratch: &mut Vec<Node>,
    verbose: bool,
    pushdown_maintain_errors: bool,
    streaming: bool,
) -> PolarsResult<()> {
    let mut stack = Vec::with_capacity(4);
    let mut names_scratch = vec![];
    let mut predicates_scratch = vec![];

    scratch.clear();
    stack.clear();

    #[derive(Default)]
    struct Value {
        // All the children of the cache per cache-id.
        children: Vec<Node>,
        parents: Vec<TwoParents>,
        cache_nodes: Vec<Node>,
        // Union over projected names.
        names_union: PlIndexSet<PlSmallStr>,
        // Union over predicates.
        predicate_union: PlIndexMap<Expr, u32>,
    }
    let mut cache_schema_and_children = PlIndexMap::new();

    // Stack frame
    #[derive(Default, Clone)]
    struct Frame {
        current: Node,
        cache_id: Option<UniqueId>,
        parent: TwoParents,
    }
    let init = Frame {
        current: root,
        ..Default::default()
    };

    stack.push(init);

    // Create the indexmap entries in topological graph order.
    ir_graph_traversal(
        root,
        &mut FnVisitors::new(
            || (),
            |key, storage: &mut IRTraversalStorage<'_>, _| {
                if let IR::Cache { input: _, id } = storage.get(key) {
                    cache_schema_and_children.insert(*id, Value::default());
                }

                ControlFlow::Continue(SubtreeVisit::Visit)
            },
            |_, _, _| ControlFlow::<()>::Continue(()),
        ),
        &mut vec![],
        &mut vec![],
        IRTraversalStorage {
            arena: lp_arena,
            skip_subtree: |_| false,
        },
    )
    .continue_value()
    .unwrap();

    // # First traversal.
    // Collect the union of columns per cache id.
    // And find the cache parents.
    while let Some(mut frame) = stack.pop() {
        let lp = lp_arena.get(frame.current);
        lp.copy_inputs(scratch);

        if let IR::Cache { input, id, .. } = lp {
            if frame.parent[0].is_some() {
                // Projection pushdown has already run and blocked on cache nodes
                // the pushed down columns are projected just above this cache
                // if there were no pushed down column, we just take the current
                // nodes schema
                // we never want to naively take parents, as a join or aggregate for instance
                // change the schema

                let v = cache_schema_and_children.get_mut(id).unwrap();
                v.children.push(*input);
                v.parents.push(frame.parent);
                v.cache_nodes.push(frame.current);

                let mut found_required_columns = false;

                for parent_node in frame.parent.into_iter().flatten() {
                    let keep_going = get_upper_projections(
                        parent_node,
                        lp_arena,
                        expr_arena,
                        &mut names_scratch,
                        &mut found_required_columns,
                    );
                    if !names_scratch.is_empty() {
                        v.names_union.extend(names_scratch.drain(..));
                    }
                    // We stop early as we want to find the first projection node above the cache.
                    if !keep_going {
                        break;
                    }
                }

                for parent_node in frame.parent.into_iter().flatten() {
                    let keep_going = get_upper_predicates(
                        parent_node,
                        lp_arena,
                        expr_arena,
                        &mut predicates_scratch,
                    );
                    if !predicates_scratch.is_empty() {
                        for pred in predicates_scratch.drain(..) {
                            let count = v.predicate_union.entry(pred).or_insert(0);
                            *count += 1;
                        }
                    }
                    // We stop early as we want to find the first predicate node above the cache.
                    if !keep_going {
                        break;
                    }
                }

                // There was no explicit projection and we must take
                // all columns
                if !found_required_columns {
                    let schema = lp.schema(lp_arena);
                    v.names_union.extend(schema.iter_names_cloned());
                }
            }
            frame.cache_id = Some(*id);
        };

        // Shift parents.
        frame.parent[1] = frame.parent[0];
        frame.parent[0] = Some(frame.current);
        for n in scratch.iter() {
            let mut new_frame = frame.clone();
            new_frame.current = *n;
            stack.push(new_frame);
        }
        scratch.clear();
    }

    // # Second pass.
    // we create a subtree where we project the columns
    // just before the cache. Then we do another projection pushdown
    // and finally remove that last projection and stitch the subplan
    // back to the cache node again
    if !cache_schema_and_children.is_empty() {
        let mut pred_pd = PredicatePushDown::new(pushdown_maintain_errors, streaming);
        // rev() the iter to visit/optimize the caches below the current cache before the current cache,
        // otherwise we get `IR::Invalid` as predicate pd `take()`s from the IR arena.
        for v in cache_schema_and_children.into_values().rev() {
            // # CHECK IF WE NEED TO REMOVE CACHES
            // If we encounter multiple distinct predicates, the caches carry different filters
            // above them (predicate pushdown was blocked by the cache nodes). Removing the caches
            // lets pushdown resume, but this only pays off if the predicates can actually be pushed
            // past the cached subplan. If a predicate refers to a column computed within the cached
            // subplan, it cannot be pushed and removing the caches would lose the subplan sharing
            // without any benefit. See #19479.
            //
            // We therefore only remove the caches if _every_ filter above them is actually pushed
            // by predicate pushdown. We probe this on cache-free copies of the subplans, buffering
            // the optimized copies. If any filter was not pushed we bail out, keeping the caches;
            // otherwise we commit the buffered copies, removing the caches.
            if v.predicate_union.len() > 1 {
                let mut replacements = Vec::with_capacity(v.cache_nodes.len());
                let mut remove_caches = true;

                for (&cache, parents) in v.cache_nodes.iter().zip(v.parents.iter()) {
                    // Restart predicate and projection pushdown from most top parent.
                    // This to ensure we continue the optimization where it was blocked initially.
                    // We pick up the blocked filter and projection.
                    let mut node = cache;
                    for p_node in parents.iter().flatten().copied() {
                        match lp_arena.get(p_node) {
                            IR::Filter { .. } | IR::SimpleProjection { .. } => true,
                            _ => break,
                        };

                        node = p_node
                    }

                    // The filter (if any) that blocked pushdown sits directly above the cache.
                    let filter_predicate = get_filter_node(*parents, lp_arena).map(|filter_node| {
                        let IR::Filter { predicate, .. } = lp_arena.get(filter_node) else {
                            unreachable!()
                        };
                        predicate.clone()
                    });

                    // Copy the subplan without caches and re-run predicate pushdown on the copy.
                    let copied_node = deep_copy_ir_delete_caches(node, lp_arena, expr_arena);
                    let lp = lp_arena.take(copied_node);
                    let lp = pred_pd.optimize(lp, lp_arena, expr_arena)?;

                    // If there is no filter or the filter was not pushed past the (former) cached
                    // subplan, removing the caches provides no benefit and we keep them.
                    let filter_pushed = filter_predicate
                        .as_ref()
                        .is_some_and(|pred| filter_was_pushed(&lp, pred, lp_arena, expr_arena));
                    if !filter_pushed {
                        remove_caches = false;
                        break;
                    }

                    replacements.push((node, lp));
                }

                if remove_caches {
                    if verbose {
                        eprintln!("cache nodes will be removed because predicates don't match")
                    }
                    for (node, lp) in replacements {
                        lp_arena.replace(node, lp);
                    }
                    continue;
                }

                // Not all filters could be pushed: keep the caches. We fall through to the regular
                // handling below, which - as `allow_parent_predicate_pushdown` is `false` for
                // multiple predicates - runs predicate pushdown on the shared subplan while leaving
                // the filters above the caches in place.
            }
            // Below we restart projection and predicates pushdown
            // on the first cache node. As it are cache nodes, the others are the same
            // and we can reuse the optimized state for all inputs.
            // See #21637

            // # RUN PREDICATE PUSHDOWN
            // Run this after projection pushdown, otherwise the predicate columns will not be projected.

            // - If all predicates of parent are the same we will restart predicate pushdown from the parent FILTER node.
            // - Otherwise we will start predicate pushdown from the cache node.
            let allow_parent_predicate_pushdown = v.predicate_union.len() == 1 && {
                let (_pred, count) = v.predicate_union.iter().next().unwrap();
                *count == v.children.len() as u32
            };

            if allow_parent_predicate_pushdown {
                let parents = *v.parents.first().unwrap();
                let node = get_filter_node(parents, lp_arena)
                    .expect("expected filter; this is an optimizer bug");
                let start_lp = lp_arena.take(node);

                let mut pred_pd =
                    PredicatePushDown::new(pushdown_maintain_errors, streaming).block_at_cache(1);
                let lp = pred_pd.optimize(start_lp, lp_arena, expr_arena)?;
                lp_arena.replace(node, lp.clone());

                let mut updated_cache_node = node;

                loop {
                    match lp_arena.get(updated_cache_node) {
                        IR::Cache { .. } => break,
                        IR::SimpleProjection { input, .. } => updated_cache_node = *input,
                        _ => unreachable!(),
                    }
                }

                for &parents in &v.parents[1..] {
                    let filter_node = get_filter_node(parents, lp_arena)
                        .expect("expected filter; this is an optimizer bug");

                    let IR::Filter { input, .. } = lp_arena.get(filter_node) else {
                        unreachable!()
                    };

                    let new_lp = match lp_arena.get(*input) {
                        IR::SimpleProjection { input, columns } => {
                            debug_assert!(matches!(lp_arena.get(*input), IR::Cache { .. }));
                            IR::SimpleProjection {
                                input: updated_cache_node,
                                columns: columns.clone(),
                            }
                        },
                        ir => {
                            debug_assert!(matches!(ir, IR::Cache { .. }));
                            lp_arena.get(updated_cache_node).clone()
                        },
                    };

                    lp_arena.replace(filter_node, new_lp);
                }
            } else {
                let child = *v.children.first().unwrap();
                let child_lp = lp_arena.take(child);
                let lp = pred_pd.optimize(child_lp, lp_arena, expr_arena)?;
                lp_arena.replace(child, lp.clone());
                for &child in &v.children[1..] {
                    lp_arena.replace(child, lp.clone());
                }
            }
        }
    }
    Ok(())
}

fn get_filter_node(parents: TwoParents, lp_arena: &Arena<IR>) -> Option<Node> {
    parents
        .into_iter()
        .flatten()
        .find(|&parent| matches!(lp_arena.get(parent), IR::Filter { .. }))
}

/// Determine whether `predicate` was pushed down when running predicate pushdown on a cache-free
/// copy of a cached subplan (rooted at `optimized_lp`).
///
/// If the predicate could not be pushed, it is re-applied as a `Filter` at the top of the subplan
/// still carrying our original predicate. In that case removing the caches buys us nothing, so we
/// return `false`. Only `SimpleProjection` nodes may end up above that filter.
fn filter_was_pushed(
    optimized_lp: &IR,
    predicate: &ExprIR,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> bool {
    // Skip any projections that pushdown may have inserted above the filter.
    let mut current = optimized_lp;
    while let IR::SimpleProjection { input, .. } = current {
        current = lp_arena.get(*input);
    }

    // The predicate was not pushed iff the top-most non-projection node is a filter that still
    // carries our original predicate.
    match current {
        IR::Filter {
            predicate: top_predicate,
            ..
        } => !expr_ir_eq(predicate, top_predicate, expr_arena),
        _ => true,
    }
}

fn expr_ir_eq(l: &ExprIR, r: &ExprIR, expr_arena: &Arena<AExpr>) -> bool {
    l.get_alias() == r.get_alias()
        && AexprNode::new(l.node()).hashable_and_cmp(expr_arena)
            == AexprNode::new(r.node()).hashable_and_cmp(expr_arena)
}
