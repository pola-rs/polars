use std::sync::Arc;

#[allow(clippy::disallowed_types)]
use polars_core::prelude::PlHashMap;
use polars_core::scalar::Scalar;
#[allow(clippy::disallowed_types)]
use polars_utils::aliases::PlHashSet;
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;

use crate::plans::iterator::ArenaExprIter;
use crate::plans::{
    AExpr, IR, IRFunctionExpr, LiteralValue,
    is_inherently_nondeterministic_excluding_udfs_top_level,
};

/// The caches in the steps of recursive queries that are computed less often than their
/// readers run: a cache is computed again in every round only if the innermost recursive query
/// around it lists it in its `round_local_caches`.
#[allow(clippy::disallowed_types)] // We don't iterate over it.
pub(super) fn caches_shared_across_rounds(root: Node, ir_arena: &Arena<IR>) -> PlHashSet<UniqueId> {
    let mut out = PlHashSet::default();
    visit_step_caches(root, ir_arena, |id, round_local| {
        if !round_local.contains(&id) {
            out.insert(id);
        }
    });
    out
}

/// The caches that are read in the step of a recursive query.
#[cfg(feature = "cse")]
#[allow(clippy::disallowed_types)] // We don't iterate over it.
pub(super) fn caches_in_recursive_steps(root: Node, ir_arena: &Arena<IR>) -> PlHashSet<UniqueId> {
    let mut out = PlHashSet::default();
    visit_step_caches(root, ir_arena, |id, _| {
        out.insert(id);
    });
    out
}

/// The innermost recursive query around a node: its id and its round-local caches.
type Recursion<'a> = (UniqueId, &'a [UniqueId]);

/// Calls `f` for every read of a cache in the step of a recursive query, with the
/// `round_local_caches` of the innermost recursive query around it.
fn visit_step_caches(root: Node, ir_arena: &Arena<IR>, mut f: impl FnMut(UniqueId, &[UniqueId])) {
    #[allow(clippy::disallowed_types)] // We don't iterate over it.
    let mut seen_caches = PlHashSet::default();
    let mut stack: Vec<(Node, Option<Recursion>)> = vec![(root, None)];
    while let Some((node, recursion)) = stack.pop() {
        match ir_arena.get(node) {
            IR::Cache { input, id } => {
                if let Some((_, round_local)) = recursion {
                    f(*id, round_local);
                }
                // The input is walked once per recursive query around it, as each can protect
                // other caches in it.
                if seen_caches.insert((*id, recursion.map(|(rec_id, _)| rec_id))) {
                    stack.push((*input, recursion));
                }
            },
            IR::Recursive {
                anchor,
                step,
                id,
                round_local_caches,
            } => {
                stack.push((*anchor, recursion));
                stack.push((*step, Some((*id, round_local_caches))));
            },
            ir => stack.extend(ir.inputs().map(|input| (input, recursion))),
        }
    }
}

/// Puts each largest part of the step of a recursive query that gives the same rows in every
/// round behind a new cache, so the query computes it once instead of once per round. Also turns
/// off the predicates that are set while the query runs in the steps, as later rounds would read
/// the values of the first round.
pub(super) fn cache_round_independent_parts(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    for (step, round_local) in recursive_steps(root, ir_arena) {
        remove_dynamic_preds(step, &round_local, ir_arena, expr_arena);
        cache_parts(step, &round_local, ir_arena, expr_arena);
    }
}

/// The step of every recursive query, outer queries first, with the round-local caches of the
/// query and of the queries around it.
fn recursive_steps(root: Node, ir_arena: &Arena<IR>) -> Vec<(Node, Vec<UniqueId>)> {
    let mut out = Vec::new();
    #[allow(clippy::disallowed_types)] // We don't iterate over it.
    let mut seen_caches = PlHashSet::default();
    let mut stack = vec![(root, Vec::new())];
    while let Some((node, round_local)) = stack.pop() {
        match ir_arena.get(node) {
            IR::Cache { input, id } => {
                if seen_caches.insert(*id) {
                    stack.push((*input, round_local));
                }
            },
            IR::Recursive {
                anchor,
                step,
                round_local_caches,
                ..
            } => {
                let mut step_round_local = round_local.clone();
                step_round_local.extend(round_local_caches.iter().copied());
                out.push((*step, step_round_local.clone()));
                stack.push((*anchor, round_local));
                stack.push((*step, step_round_local));
            },
            ir => stack.extend(ir.inputs().map(|input| (input, round_local.clone()))),
        }
    }
    out
}

/// Turns off the predicates that are set while the query runs in a step, up to the caches that
/// are computed once and the steps of the recursive queries in it.
fn remove_dynamic_preds(
    step: Node,
    round_local: &[UniqueId],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let mut stack = vec![step];
    let mut dynamic_preds = vec![];
    while let Some(node) = stack.pop() {
        let ir = ir_arena.get(node);
        for e in ir.exprs() {
            dynamic_preds.extend(expr_arena.iter(e.node()).filter_map(|(n, ae)| {
                matches!(
                    ae,
                    AExpr::Function {
                        function: IRFunctionExpr::DynamicPred { .. },
                        ..
                    }
                )
                .then_some(n)
            }));
        }
        match ir {
            IR::Cache { input, id } if round_local.contains(id) => stack.push(*input),
            IR::Cache { .. } => {},
            IR::Recursive { anchor, .. } => stack.push(*anchor),
            ir => stack.extend(ir.inputs()),
        }
        match ir_arena.get_mut(node) {
            IR::Sort {
                slice: Some((_, _, pred)),
                ..
            } => *pred = None,
            IR::Join { options, .. } if !options.runtime_filters.is_empty() => {
                Arc::make_mut(options).runtime_filters.clear()
            },
            _ => {},
        }
    }
    for node in dynamic_preds {
        expr_arena.replace(
            node,
            AExpr::Literal(LiteralValue::Scalar(Scalar::from(true))),
        );
    }
}

/// Puts each largest part of `step` that gives the same rows in every round behind a new cache.
fn cache_parts(
    step: Node,
    round_local: &[UniqueId],
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) {
    // Whether a node must run in every round: it reads a round, or it may give other rows when
    // it runs again.
    #[allow(clippy::disallowed_types)] // We don't iterate over it.
    let mut per_round = PlHashMap::default();
    let mut stack = vec![(step, false)];
    while let Some((node, inputs_done)) = stack.pop() {
        if per_round.contains_key(&node) {
            continue;
        }
        let ir = ir_arena.get(node);
        let value = match ir {
            IR::WorkingTableScan { .. } | IR::Recursive { .. } => true,
            IR::Cache { id, .. } if !round_local.contains(id) => false,
            _ if !inputs_done => {
                stack.push((node, true));
                stack.extend(ir.inputs().map(|input| (input, false)));
                continue;
            },
            // A round-local cache may be read by parts that read the round.
            IR::Cache { .. } => true,
            _ => is_volatile(ir, expr_arena) || ir.inputs().any(|input| per_round[&input]),
        };
        per_round.insert(node, value);
    }

    let mut stack = vec![step];
    #[allow(clippy::disallowed_types)] // We don't iterate over it.
    let mut seen = PlHashSet::default();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        let ir = ir_arena.get(node);
        if per_round[&node] {
            if !matches!(ir, IR::Recursive { .. }) {
                stack.extend(ir.inputs());
            }
            continue;
        }
        if matches!(ir, IR::Cache { .. } | IR::DataFrameScan { .. }) {
            continue;
        }
        let ir = ir_arena.take(node);
        let input = ir_arena.add(ir);
        ir_arena.replace(
            node,
            IR::Cache {
                input,
                id: UniqueId::new(),
            },
        );
    }
}

/// Whether running `ir` again may give other rows, as with random draws. Like common subplan
/// elimination, this takes user functions to give the same rows.
fn is_volatile(ir: &IR, expr_arena: &Arena<AExpr>) -> bool {
    #[cfg(feature = "python")]
    if let IR::PythonScan { options } = ir
        && !options.is_pure
    {
        return true;
    }
    ir.exprs().any(|e| {
        expr_arena
            .iter(e.node())
            .any(|(_, ae)| is_inherently_nondeterministic_excluding_udfs_top_level(ae))
    })
}
