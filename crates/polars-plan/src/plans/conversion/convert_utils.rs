use super::*;
use crate::plans::aexpr::or_factoring;

/// Lift conjuncts shared across every OR branch to top-level ANDs
/// (`(A∧X) ∨ (A∧Y) → A ∧ (X∨Y)`), so the `MintermIter` walk in
/// `SplitPredicates::new` can split them into independent `IR::Filter`
/// nodes for predicate-pushdown to route.
pub(super) fn canonicalize_predicate(predicate: Node, expr_arena: &mut Arena<AExpr>) {
    or_factoring::factor_or_in_aexpr(predicate, expr_arena);
}

/// Split expression that are ANDed into multiple Filter nodes as the optimizer can then
/// push them down independently. Especially if they refer columns from different tables
/// this will be more performant.
///
/// So:
/// * `filter[foo == bar & ham == spam]`
///
/// Becomes:
/// * `filter [foo == bar]`
/// * `filter [ham == spam]`
pub(super) struct SplitPredicates {
    pub(super) pushable: Vec<Node>,
    pub(super) fallible: Option<Node>,
}

impl SplitPredicates {
    /// Walk `predicate`'s AND chain via `MintermIter`, route each conjunct
    /// through `ExprPushdownGroup`, and bucket into `pushable` / `fallible`.
    /// Returns `None` if any conjunct classifies as `Barrier` (caller falls
    /// back to a single un-split Filter).
    ///
    /// Assumes `predicate` has been canonicalized upstream by
    /// `canonicalize_predicate`. This function only splits.
    pub(super) fn new(
        predicate: Node,
        expr_arena: &mut Arena<AExpr>,
        scratch: Option<&mut UnitVec<Node>>,
        maintain_errors: bool,
    ) -> Option<Self> {
        let mut local_scratch = unitvec![];
        let scratch = scratch.unwrap_or(&mut local_scratch);

        let mut pushable = vec![];
        let mut acc_fallible = unitvec![];

        for predicate in MintermIter::new(predicate, expr_arena) {
            use ExprPushdownGroup::*;

            let ae = expr_arena.get(predicate);

            match ExprPushdownGroup::Pushable.update_with_expr_rec(ae, expr_arena, Some(scratch)) {
                Pushable => pushable.push(predicate),

                Fallible => {
                    if maintain_errors {
                        return None;
                    }

                    acc_fallible.push(predicate);
                },

                Barrier => return None,
            }
        }

        let fallible = (!acc_fallible.is_empty()).then(|| {
            let mut node = acc_fallible.pop().unwrap();

            for next_node in acc_fallible.iter() {
                node = expr_arena.add(AExpr::BinaryExpr {
                    left: node,
                    op: Operator::And,
                    right: *next_node,
                })
            }

            node
        });

        let out = Self { pushable, fallible };

        Some(out)
    }
}
