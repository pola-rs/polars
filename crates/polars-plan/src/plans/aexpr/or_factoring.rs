//! OR factoring: `(A∧X) ∨ (A∧Y) → A ∧ (X∨Y)`
//!
//! Pure AExpr rewrite, in place on `expr_arena`. Called from
//! `simplify_predicate` before `SplitPredicates::new` walks the AND
//! chain; factored commons become AND-conjuncts at the top, get split
//! into their own `IR::Filter` nodes, and become visible to
//! predicate-pushdown.
//!
//! Fallibility / evaluation-order: changes the set of conjuncts seen by
//! the `Pushable | Fallible | Barrier` classifier in `SplitPredicates::new`,
//! which may change the order in which a fallible expression is evaluated
//! relative to other Filters or joins. Inherits Polars's AND-splitting
//! trade-off: final result is unchanged, error-reporting order may differ
//! as it would with a `.filter(...).filter(...)` chain.

use polars_utils::aliases::{InitHashMaps, PlIndexMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::scratch_vec::ScratchVec;

use crate::plans::aexpr::{AExpr, CanonicalExprId, CanonicalExprMap, MintermIter};
use crate::prelude::Operator;

/// Walk the AExpr tree bottom-up, applying OR factoring at every OR node.
/// Descends into every variant (not just AND/OR) so ORs nested under `Not`,
/// `Function`, `Cast`, etc. are found.
///
/// Sound under Kleene K3: distributivity of AND over OR holds for null
/// operands by case analysis.
pub(crate) fn factor_or_in_aexpr(node: Node, expr_arena: &mut Arena<AExpr>) {
    // Iterative pre-order into a Vec, then iterate reversed so descendants
    // are visited before ancestors. Matches `MintermIter`.
    let mut work = vec![node];
    let mut pre_order = Vec::new();
    while let Some(n) = work.pop() {
        pre_order.push(n);
        expr_arena.get(n).children_rev(&mut work);
    }

    let mut canonical_exprs = CanonicalExprMap::new();

    // Iterate in post-order
    for &n in pre_order.iter().rev() {
        if matches!(
            expr_arena.get(n),
            AExpr::BinaryExpr {
                op: Operator::Or | Operator::LogicalOr,
                ..
            }
        ) {
            if let Some(factored) = try_factor_or(n, &mut canonical_exprs, expr_arena) {
                // Fine to remove because we have not visited the parent of `n` yet
                canonical_exprs.remove(n, expr_arena);
                expr_arena.replace(n, factored);
            }
        }
    }
}

/// Collect an expression's OR-tree branches into `out` (dual of `MintermIter`).
fn collect_or_branches(root: Node, expr_arena: &Arena<AExpr>, out: &mut Vec<Node>) {
    let mut stack = vec![root];
    while let Some(top) = stack.pop() {
        match expr_arena.get(top) {
            AExpr::BinaryExpr {
                left,
                op: Operator::Or | Operator::LogicalOr,
                right,
            } => {
                stack.push(*right);
                stack.push(*left);
            },
            _ => out.push(top),
        }
    }
}

/// Try OR factoring on a node known to be an OR. Returns the rewritten
/// top-level `AExpr` on success.
fn try_factor_or(
    or_node: Node,
    canonical_exprs: &mut CanonicalExprMap,
    expr_arena: &mut Arena<AExpr>,
) -> Option<AExpr> {
    let mut branches = Vec::new();
    collect_or_branches(or_node, expr_arena, &mut branches);
    if branches.len() < 2 {
        return None;
    }

    let mut branch_terms: Vec<Vec<Node>> = branches
        .iter()
        .map(|&b| MintermIter::new(b, expr_arena).collect::<Vec<_>>())
        .collect();

    // Iterate the shortest branch as candidates: max commons is bounded by
    // the smallest branch, so fewer candidates means fewer wasted scans.
    // OR is commutative, so the sorted branch order is semantically identical.
    branch_terms.sort_by_key(|terms| terms.len());

    let branch_ids: Vec<Vec<CanonicalExprId>> = branch_terms
        .iter()
        .map(|terms| {
            terms
                .iter()
                .map(|&term| canonical_exprs.resolve(term, expr_arena))
                .collect()
        })
        .collect();
    let buckets: Vec<PlIndexMap<CanonicalExprId, Vec<usize>>> = branch_ids
        .iter()
        .map(|ids| {
            let mut m: PlIndexMap<CanonicalExprId, Vec<usize>> =
                PlIndexMap::with_capacity(ids.len());
            for (i, &id) in ids.iter().enumerate() {
                m.entry(id).or_default().push(i);
            }
            m
        })
        .collect();

    // `taken[b][i]` = "term i of branch b already claimed".
    let mut common = Vec::new();
    let mut taken: Vec<_> = branch_terms.iter().map(|t| vec![false; t.len()]).collect();
    let mut other_matches: ScratchVec<usize> = ScratchVec::default();

    for (cand_idx, &cand) in branch_terms[0].iter().enumerate() {
        let cand_id = branch_ids[0][cand_idx];

        // Skip inherently non-deterministic candidates: factoring them out of
        // `(A ∧ X) ∨ (A ∧ Y) → A ∧ (X ∨ Y)` would evaluate `A` once instead of
        // twice per row, which is unsound when the two evaluations could disagree.
        if canonical_exprs.is_nondeterministic(cand_id) {
            continue;
        }

        let other_matches = other_matches.get();
        let all_matched = (1..branch_terms.len()).all(|b_idx| {
            let Some(&m) = buckets[b_idx]
                .get(&cand_id)
                .and_then(|ixs| ixs.iter().find(|&i| !taken[b_idx][*i]))
            else {
                return false;
            };
            other_matches.push(m);
            true
        });
        if !all_matched {
            continue;
        }

        common.push(cand);
        taken[0][cand_idx] = true;
        for (offset, &m) in other_matches.iter().enumerate() {
            taken[offset + 1][m] = true;
        }
    }

    if common.is_empty() {
        return None;
    }

    // Rebuild branches without claimed conjuncts. `collect::<Option<_>>`
    // short-circuits when any branch becomes empty; `None` then signals
    // absorption in the match below (the OR drops out entirely).
    let residuals: Option<Vec<Node>> = branch_terms
        .into_iter()
        .enumerate()
        .map(|(b_idx, terms)| {
            let kept: Vec<_> = terms
                .into_iter()
                .enumerate()
                .filter_map(|(i, n)| (!taken[b_idx][i]).then_some(n))
                .collect();
            (!kept.is_empty()).then(|| combine_with(kept, Operator::And, expr_arena))
        })
        .collect();

    // Final shape: left-skewed AND(common..., OR(residuals...)). If the OR
    // collapsed (some branch went empty), absorption (`X ∨ (X ∧ A) → X`)
    // drops the OR entirely: result is just the AND of commons.
    let folded_node = match residuals {
        Some(nodes) => {
            let or_node = combine_with(nodes, Operator::Or, expr_arena);
            let mut all_nodes = common;
            all_nodes.push(or_node);
            combine_with(all_nodes, Operator::And, expr_arena)
        },
        None => combine_with(common, Operator::And, expr_arena),
    };
    Some(expr_arena.get(folded_node).clone())
}

/// Left-fold a non-empty stream of nodes with `op`.
fn combine_with(
    nodes: impl IntoIterator<Item = Node>,
    op: Operator,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    nodes
        .into_iter()
        .reduce(|left, right| expr_arena.add(AExpr::BinaryExpr { left, op, right }))
        .expect("combine_with: non-empty iterator")
}
