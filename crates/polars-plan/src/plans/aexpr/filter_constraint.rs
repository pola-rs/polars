//! Combine per-column comparisons in an `AND` chain: spot impossible filters
//! (replace with `false`, e.g. `a.is_in([1, 2]) AND a.is_in([3, 4])`) and drop
//! redundant comparisons (`a >= 1 AND a >= 5` to `a >= 5`; `a >= 3 AND a != 3`
//! to `a > 3`).
//!
//! `merge_filter_constraints` runs four data-dependent stages (no reordering):
//! 1. **Collect** (`classify_into_constraints`): fold each `col op lit` into a
//!    per-column `ColumnConstraints`; unmodeled conjuncts kept aside.
//! 2. **Propagate** (`propagate_equalities`): carry constraints across
//!    `col == col` (`a == b AND a > 5` gives `b > 5`); needs all bounds first.
//! 3. **Resolve** (`resolve_deferred`): resolve `!=` / `is_in` / `!is_in` against
//!    the now final bounds; needs propagated bounds, so runs after stage 2.
//! 4. **Emit**: `false` if impossible (gated on `maintain_errors`), else the
//!    tightest rebuilt chain, or `None` if unchanged.

use std::cmp::Ordering;

use polars_core::prelude::AnyValue;
#[cfg(feature = "is_in")]
use polars_core::prelude::Series;
use polars_core::scalar::Scalar;
use polars_utils::aliases::{InitHashMaps, PlIndexMap, PlIndexSet};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::properties::ExprPushdownGroup;
use super::{AExpr, IRBooleanFunction, IRFunctionExpr, LiteralValue, MintermIter, Operator};
#[cfg(feature = "is_between")]
use crate::prelude::ClosedInterval;

// How a single conjunct in the `AND` chain was classified.
enum Classification {
    // Impossible during the walk: bounds crossed (empty range), or conflicting
    // nullability requirements (`is_null` vs `is_not_null`, or vs a comparison's
    // implied non-null). Other impossibilities surface later (not as this variant):
    // cross-column conflicts in `propagate_equalities`; `!=` / `is_in` / `!is_in` /
    // `is_null` with a comparison in `resolve_deferred`.
    Unsat,
    // A bound folded into its column and rebuilt from the merged constraints,
    // so the original conjunct is dropped.
    Bound,
    // Like `Bound`, but the conjunct was a negation normalized into positive form
    // (`!(x > 5)` to `x <= 5`). Forces a rewrite even when nothing tightens, since
    // the original `!` node is gone (and a positive `col op lit` prunes where a
    // `!` node cannot).
    Normalized,
    // A `col == col` equality, whose two columns form a cross-column propagation
    // edge (the node is kept verbatim unless its class becomes fixed).
    Equality(PlSmallStr, PlSmallStr),
    // Node kept verbatim: either unmodeled (UDFs, null-aware ops, arithmetic) or
    // modeled only for contradiction (`is_in` / `!is_in` / `is_null` / `is_not_null` /
    // non-`==` column-column comparisons record but aren't rewritten).
    Opaque,
}

// How a `!= value` exclusion resolves once the bounds are known.
enum ExcludedResolution {
    // Outside the feasible range, so the exclusion is redundant.
    Drop,
    // Strictly inside the range, so it survives as a `!= value`.
    Keep,
    // On an inclusive lower/upper endpoint, so it tightens that bound to exclusive.
    TightenLower,
    TightenUpper,
}

// What `is_null` / `is_not_null` require of the surviving rows. A value
// constraint (bound, `!=`, `is_in`, `!is_in`) drops nulls under K3, so any of them
// implies the column is `NonNull` and contradicts a `Null` requirement.
#[derive(Clone, Copy, Default, PartialEq)]
enum Nullability {
    #[default]
    Unconstrained,
    Null,
    NonNull,
}

// Orders two scalars; incomparable types give `None` (neither equal nor ordered),
// so we never declare a contradiction we aren't sure of.
fn scalar_cmp(a: &Scalar, b: &Scalar) -> Option<Ordering> {
    a.as_any_value().partial_cmp(&b.as_any_value())
}

fn scalar_eq(a: &Scalar, b: &Scalar) -> bool {
    scalar_cmp(a, b) == Some(Ordering::Equal)
}

// `PlIndexSet` (not a hash set) keeps insertion order so the rebuilt conjuncts stay
// deterministic.
#[derive(Clone, Default)]
struct ColumnConstraints {
    // `bool` = inclusive (`>=`/`<=` vs `>`/`<`). `== x` is an inclusive `[x, x]`.
    lower: Option<(Scalar, bool)>,
    upper: Option<(Scalar, bool)>,
    // `!= x` values, resolved against the bounds in `resolve_deferred`.
    excluded: PlIndexSet<Scalar>,
    // Intersection of all `is_in` haystacks; resolved against bounds in `resolve_deferred`.
    allowed: Option<PlIndexSet<Scalar>>,
    // Union of all `!is_in` haystacks; the dual of `allowed`. Used for contradiction
    // detection only, never emitted (a large `!is_in` must not explode into N `!=`).
    disallowed: PlIndexSet<Scalar>,
    // Nullability required by `is_null` / `is_not_null` on this column.
    nullability: Nullability,
    // Empty value set (unsatisfiable). Set by any stage (crossed bounds,
    // `!=`/`is_in`/`!is_in`, cross-column conflict, null/value conflict) and sticky,
    // so `add_*` bail early once set.
    unsat: bool,
}

impl ColumnConstraints {
    // Keep the tighter bound, returning whether the slot changed. `tighter` is the
    // new-vs-existing ordering that means the new bound wins (`Greater` lower, `Less` upper).
    fn tighten(
        slot: &mut Option<(Scalar, bool)>,
        value: Scalar,
        inclusive: bool,
        tighter: Ordering,
    ) -> bool {
        match slot.take() {
            None => {
                *slot = Some((value, inclusive));
                true
            },
            Some((existing, existing_inclusive)) => match scalar_cmp(&value, &existing) {
                // Strictly tighter value wins; the bound always changes.
                Some(ord) if ord == tighter => {
                    *slot = Some((value, inclusive));
                    true
                },
                // Same value: an exclusive bound is tighter than an inclusive one.
                Some(Ordering::Equal) => {
                    let new_inclusive = inclusive && existing_inclusive;
                    let changed = new_inclusive != existing_inclusive;
                    *slot = Some((existing, new_inclusive));
                    changed
                },
                // Less tight, or incomparable types: keep the existing bound.
                _ => {
                    *slot = Some((existing, existing_inclusive));
                    false
                },
            },
        }
    }

    fn add_lower(&mut self, value: Scalar, inclusive: bool) -> bool {
        if self.unsat {
            return false;
        }
        let changed = Self::tighten(&mut self.lower, value, inclusive, Ordering::Greater);
        self.check_bound_consistency();
        changed
    }

    fn add_upper(&mut self, value: Scalar, inclusive: bool) -> bool {
        if self.unsat {
            return false;
        }
        let changed = Self::tighten(&mut self.upper, value, inclusive, Ordering::Less);
        self.check_bound_consistency();
        changed
    }

    fn add_excluded(&mut self, value: Scalar) -> bool {
        if self.unsat {
            return false;
        }
        // `insert` dedups, so repeated `!= x` collapse to one.
        self.excluded.insert(value)
    }

    #[cfg(feature = "is_in")]
    fn add_allowed(&mut self, values: PlIndexSet<Scalar>) -> bool {
        if self.unsat {
            return false;
        }
        match &mut self.allowed {
            None => {
                self.allowed = Some(values);
                true
            },
            // A second `is_in` intersects: keep only values present in both.
            Some(existing) => {
                let before = existing.len();
                existing.retain(|e| values.contains(e));
                existing.len() != before
            },
        }
    }

    // Adds a `!is_in` haystack to the disallowed set (union: a value is disallowed
    // if any `!is_in` excludes it).
    #[cfg(feature = "is_in")]
    fn add_disallowed(&mut self, values: impl IntoIterator<Item = Scalar>) -> bool {
        if self.unsat {
            return false;
        }
        let before = self.disallowed.len();
        self.disallowed.extend(values);
        self.disallowed.len() != before
    }

    fn has_value_constraint(&self) -> bool {
        self.lower.is_some()
            || self.upper.is_some()
            || !self.excluded.is_empty()
            || self.allowed.is_some()
            || !self.disallowed.is_empty()
    }

    // Records an `is_null` (`Null`) / `is_not_null` (`NonNull`) requirement. The first
    // wins; the opposite requirement makes the column impossible. The `Null`-vs-value
    // contradiction is checked once in `resolve_deferred` (a value drops nulls).
    fn require_nullability(&mut self, required: Nullability) {
        debug_assert!(
            required != Nullability::Unconstrained,
            "callers require Null or NonNull"
        );
        if self.nullability == Nullability::Unconstrained {
            self.nullability = required;
        } else if self.nullability != required {
            self.unsat = true; // opposite requirement: `is_null` AND `is_not_null`
        }
    }

    // Intersects another column's constraints into this one (for `col == col`: on a
    // surviving row the two are equal, so each bound/exclusion/`is_in`/`!is_in` holds
    // for both). Returns `(emitted, any)` change flags. The split matters: bounds and
    // `!=` feed the rebuilt chain, so they may trigger a rewrite; `is_in`/`!is_in`
    // only detect contradictions and are never emitted, so counting them as a rewrite
    // trigger would re-fire the rule on its own (unchanged) output every pass, and
    // the optimizer would never converge. They still drive the propagation fixpoint
    // via `any` so transitive chains complete.
    fn merge_from(&mut self, other: &ColumnConstraints) -> (bool, bool) {
        if self.unsat {
            return (false, false);
        }
        let mut emitted = false;
        let mut any = false;
        if let Some((value, inclusive)) = &other.lower {
            emitted |= self.add_lower(value.clone(), *inclusive);
        }
        if let Some((value, inclusive)) = &other.upper {
            emitted |= self.add_upper(value.clone(), *inclusive);
        }
        for value in &other.excluded {
            emitted |= self.add_excluded(value.clone());
        }
        #[cfg(feature = "is_in")]
        if let Some(values) = &other.allowed {
            any |= self.add_allowed(values.clone());
        }
        #[cfg(feature = "is_in")]
        {
            any |= self.add_disallowed(other.disallowed.iter().cloned());
        }
        any |= emitted;
        (emitted, any)
    }

    // Marks the column impossible when the lower bound exceeds the upper (empty range).
    fn check_bound_consistency(&mut self) {
        if self.unsat {
            return;
        }
        if let (Some((lo, lo_inc)), Some((hi, hi_inc))) = (&self.lower, &self.upper) {
            match scalar_cmp(lo, hi) {
                Some(Ordering::Greater) => self.unsat = true,
                Some(Ordering::Equal) if !(*lo_inc && *hi_inc) => self.unsat = true,
                _ => {},
            }
        }
    }

    // Resolves the `!=` exclusions and `is_in` set against the now-known bounds. An
    // exclusion on an inclusive endpoint tightens it (`x >= 3 AND x != 3` to `x > 3`),
    // one outside the range drops, one inside stays; may make the column impossible.
    // Deliberately does NOT do discrete-successor tightening (`x > 2 AND x != 3` to
    // `x >= 4` for integer dtypes): that needs per-dtype successor arithmetic with
    // overflow handling at the type bounds plus sorted exclusions, left as follow-up.
    fn resolve_deferred(&mut self) {
        // A value constraint drops nulls, so it can't coexist with an `is_null`
        // requirement (`a.is_null() AND a > 5`). Checked here so it holds regardless
        // of the order the conjuncts arrived in. Soundness rests on every source of a
        // value constraint being null-dropping under K3 (bounds, `!=`, `is_in`,
        // `!is_in`); the null-*preserving* ops `eq_missing` / `ne_missing`
        // (`EqValidity` / `NotEqValidity`) must therefore stay `Opaque` and never feed
        // `has_value_constraint`, or `a.is_null() AND a.ne_missing(4)` would wrongly
        // collapse (it's satisfiable on the null rows).
        if self.nullability == Nullability::Null && self.has_value_constraint() {
            self.unsat = true;
            return;
        }

        for key in std::mem::take(&mut self.excluded) {
            if self.unsat {
                break;
            }
            match self.classify_excluded(&key) {
                ExcludedResolution::Drop => {},
                ExcludedResolution::Keep => {
                    self.excluded.insert(key);
                },
                ExcludedResolution::TightenLower => self.lower = Some((key, false)),
                ExcludedResolution::TightenUpper => self.upper = Some((key, false)),
            }
            self.check_bound_consistency();
        }

        // `is_in` confines the column to a finite set; impossible if no member is
        // feasible (disjoint `is_in`, or `is_in([1, 2]) AND a != 1 AND a != 2`). The
        // `is_feasible` check also rejects any `!is_in`-disallowed value.
        if !self.unsat {
            if let Some(allowed) = &self.allowed {
                if !allowed.iter().any(|v| self.is_feasible(v)) {
                    self.unsat = true;
                }
            }
        }

        // A value pinned by `== v` and also `!is_in`-disallowed is impossible
        // (`x == 2 AND !x.is_in([2])`). The `allowed` scan above covers the
        // `is_in`-confined case; this covers the bound-pinned one.
        if !self.unsat {
            if let Some(v) = self.fixed_value() {
                if self.disallowed.contains(v) {
                    self.unsat = true;
                }
            }
        }
    }

    // Whether `value` survives the bounds, exclusions, and `is_in` set. Incomparable
    // types count as feasible, so we never declare a contradiction we aren't sure of.
    fn is_feasible(&self, value: &Scalar) -> bool {
        if let Some((lo, lo_inc)) = &self.lower {
            match scalar_cmp(value, lo) {
                Some(Ordering::Less) => return false,
                Some(Ordering::Equal) if !lo_inc => return false,
                _ => {},
            }
        }
        if let Some((hi, hi_inc)) = &self.upper {
            match scalar_cmp(value, hi) {
                Some(Ordering::Greater) => return false,
                Some(Ordering::Equal) if !hi_inc => return false,
                _ => {},
            }
        }
        if self.excluded.contains(value) || self.disallowed.contains(value) {
            return false;
        }
        // Confined to the `is_in` set when one is present.
        self.allowed
            .as_ref()
            .is_none_or(|allowed| allowed.contains(value))
    }

    // The single value the column is pinned to (`== v`, an inclusive `[v, v]`), if any.
    fn fixed_value(&self) -> Option<&Scalar> {
        match (&self.lower, &self.upper) {
            (Some((lo, true)), Some((hi, true))) if scalar_eq(lo, hi) => Some(lo),
            _ => None,
        }
    }

    fn classify_excluded(&self, value: &Scalar) -> ExcludedResolution {
        if let Some((lo, lo_inc)) = &self.lower {
            match scalar_cmp(value, lo) {
                Some(Ordering::Less) => return ExcludedResolution::Drop,
                Some(Ordering::Equal) => {
                    return if *lo_inc {
                        ExcludedResolution::TightenLower
                    } else {
                        ExcludedResolution::Drop
                    };
                },
                _ => {},
            }
        }
        if let Some((hi, hi_inc)) = &self.upper {
            match scalar_cmp(value, hi) {
                Some(Ordering::Greater) => return ExcludedResolution::Drop,
                Some(Ordering::Equal) => {
                    return if *hi_inc {
                        ExcludedResolution::TightenUpper
                    } else {
                        ExcludedResolution::Drop
                    };
                },
                _ => {},
            }
        }
        ExcludedResolution::Keep
    }
}

/// Rewrites `predicate`'s `AND` chain to a tighter equivalent, or `None` if
/// nothing changes. Either collapses to `Literal(false)` when the comparisons
/// can't all hold (letting the filter become an empty scan), or merges redundant
/// per-column comparisons into the tightest set (`a >= 1 AND a >= 5` to `a >= 5`,
/// `a >= 3 AND a != 3` to `a > 3`).
pub(crate) fn merge_filter_constraints(
    predicate: Node,
    maintain_errors: bool,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    // Collect bounds per column; unmodeled conjuncts kept aside, `col == col`
    // routed to `edges` for cross-column propagation. Stop early once a column is
    // impossible (the whole filter is then false).
    let mut constraints: PlIndexMap<PlSmallStr, ColumnConstraints> = PlIndexMap::new();
    let mut opaque: Vec<Node> = Vec::new();
    let mut edges: Vec<(PlSmallStr, PlSmallStr, Node)> = Vec::new();
    let mut num_bound_conjuncts = 0usize;
    let mut normalized = false;
    let mut unsat = false;

    for conjunct in MintermIter::new(predicate, expr_arena) {
        match classify_into_constraints(expr_arena.get(conjunct), expr_arena, &mut constraints) {
            Classification::Unsat => {
                unsat = true;
                break;
            },
            Classification::Bound => num_bound_conjuncts += 1,
            Classification::Normalized => {
                num_bound_conjuncts += 1;
                normalized = true;
            },
            Classification::Equality(a, b) => edges.push((a, b, conjunct)),
            Classification::Opaque => opaque.push(conjunct),
        }
    }

    // Cross-column propagation: merge each equality class's constraints across its
    // members (`a == b AND a > 5` gives `b > 5`). Before `resolve_deferred`, so
    // `!=`/`is_in` resolve against any propagated bounds.
    let mut propagated = false;
    let mut dropped_equality = false;
    // No `col == col` edges means nothing to propagate.
    if !unsat && !edges.is_empty() {
        propagated = propagate_equalities(&mut constraints, &edges);

        // Drop `a == b` once its class is fixed (both ends become `== v`, redundant);
        // otherwise keep it among the opaque conjuncts for the rebuild.
        for (a, _, node) in &edges {
            if constraints
                .get(a)
                .and_then(ColumnConstraints::fixed_value)
                .is_some()
            {
                dropped_equality = true;
            } else {
                opaque.push(*node);
            }
        }
    }

    // Resolve `!=`/`is_in` against the final bounds (deferred since it needs them,
    // including propagated values, and can newly make a column impossible). `any`
    // is lazy, so it stops at the first impossible column.
    if !unsat {
        unsat = constraints.values_mut().any(|cc| {
            cc.resolve_deferred();
            cc.unsat
        });
    }

    if unsat {
        // Collapsing to `false` drops the filter, so an expression that would have
        // errored no longer runs; only do it when errors needn't be kept (same rule
        // as predicate pushdown). No determinism check needed: contradictions are
        // decided only by bare `col`/`lit` shapes (comparisons, `is_in`, null checks),
        // and a plain column is deterministic.
        let mut group = ExprPushdownGroup::Pushable;
        group.update_with_expr_rec(expr_arena.get(predicate), expr_arena, None);
        if group.blocks_pushdown(maintain_errors) {
            return None;
        }
        return Some(expr_arena.add(AExpr::Literal(Scalar::from(false).into())));
    }

    // Satisfiable: collect the tightest comparisons per column. Dropping a redundant
    // `col op lit` is sound regardless of `maintain_errors` (each is pure/infallible
    // and every other conjunct is kept). Built arena-free, so the common no-op path
    // adds no nodes.
    let mut comparisons: Vec<(&PlSmallStr, Operator, &Scalar)> = Vec::new();
    for (name, cc) in &constraints {
        collect_column_comparisons(name, cc, &mut comparisons);
    }

    // Rewrite only if we tightened, normalized a negation, propagated, or dropped an
    // `a == b`; else leave the plan alone (no churn, no optimizer loop).
    let tightened = comparisons.len() < num_bound_conjuncts;
    if !(tightened || normalized || propagated || dropped_equality) {
        return None;
    }

    // Materialize the nodes (the only `Scalar` clones) now that we know we rewrite.
    for (name, op, value) in comparisons {
        opaque.push(comparison_node(name, op, value.clone(), expr_arena));
    }
    Some(fold_and(opaque, expr_arena))
}

// Merges constraints across each `col == col` edge both ways, to a fixpoint so
// transitive chains (`a == b AND b == c`) propagate. Carries bounds (`a > 5` to
// `b > 5`), folds equal fixed values, and flags conflicts impossible via the
// bound-consistency check (`a == b AND a == 5 AND b == 6`). Each merge only
// tightens, so it terminates. Returns whether anything *emittable* (a bound or
// `!=`) was propagated; see `merge_from` for why `is_in`/`!is_in` propagation
// advances the fixpoint but must not report as a change.
fn propagate_equalities(
    constraints: &mut PlIndexMap<PlSmallStr, ColumnConstraints>,
    edges: &[(PlSmallStr, PlSmallStr, Node)],
) -> bool {
    let mut emitted_changed = false;
    loop {
        let mut progressed = false;
        for (a, b, _) in edges {
            if a == b {
                continue; // self-edge (`col == col` on one column): nothing to merge
            }
            // Borrow both ends at once so neither needs cloning. Merging `b ← a` then
            // `a ← b` matches snapshotting both first: `a`'s pre-state is already in
            // `a`, so the second merge adds only what `b` originally held.
            let [Some(ca), Some(cb)] = constraints.get_disjoint_mut([a, b]) else {
                continue;
            };
            let (emitted_into_b, any_into_b) = cb.merge_from(ca);
            let (emitted_into_a, any_into_a) = ca.merge_from(cb);
            emitted_changed |= emitted_into_b || emitted_into_a;
            progressed |= any_into_b || any_into_a;
        }
        if !progressed {
            return emitted_changed;
        }
    }
}

// Classifies one conjunct, folding any `col op lit` bound into its column.
// A leading `!(...)` is pushed inward (`classify_negation`). Unmodeled shapes are
// reported as `Opaque` and kept verbatim by the caller; `is_in` / `is_null` /
// `is_not_null` are also `Opaque` but record (haystack / nullability) for
// contradiction detection.
fn classify_into_constraints(
    ae: &AExpr,
    expr_arena: &Arena<AExpr>,
    constraints: &mut PlIndexMap<PlSmallStr, ColumnConstraints>,
) -> Classification {
    match ae {
        AExpr::BinaryExpr { left, op, right } => {
            classify_comparison(*left, *op, *right, expr_arena, constraints)
        },
        AExpr::Function {
            input,
            function,
            options: _,
        } => {
            // Only a boolean function can be a predicate we model; anything else opaque.
            let IRFunctionExpr::Boolean(func) = function else {
                return Classification::Opaque;
            };
            match func {
                // `!(...)`: push the negation inward over the shapes we model (a
                // comparison, `is_in`, `is_null` / `is_not_null`).
                IRBooleanFunction::Not => {
                    classify_negation(input[0].node(), expr_arena, constraints)
                },
                #[cfg(feature = "is_between")]
                IRBooleanFunction::IsBetween { closed } => {
                    assert_eq!(
                        input.len(),
                        3,
                        "is_between has 3 inputs: value, lower, upper"
                    );
                    classify_is_between(
                        input[0].node(),
                        input[1].node(),
                        input[2].node(),
                        *closed,
                        expr_arena,
                        constraints,
                    )
                },
                // Record the haystack as an allowed-set for contradiction detection
                // only; keep the `is_in` node (we don't rewrite it, its null handling
                // is subtle).
                #[cfg(feature = "is_in")]
                IRBooleanFunction::IsIn { .. } => {
                    if let Some(col_name) = as_column(expr_arena.get(input[0].node())) {
                        if let Some(values) = as_value_set(expr_arena.get(input[1].node())) {
                            let allowed = values.into_iter().collect();
                            constraints
                                .entry(col_name)
                                .or_default()
                                .add_allowed(allowed);
                        }
                    }
                    Classification::Opaque
                },
                // `is_null` / `is_not_null` record a nullability requirement for
                // contradiction detection (`a.is_null() AND a > 5` is impossible since
                // any comparison drops nulls); the node is kept verbatim.
                IRBooleanFunction::IsNull => {
                    classify_null(input[0].node(), Nullability::Null, expr_arena, constraints)
                },
                IRBooleanFunction::IsNotNull => classify_null(
                    input[0].node(),
                    Nullability::NonNull,
                    expr_arena,
                    constraints,
                ),
                _ => Classification::Opaque,
            }
        },
        // Not a `col op lit`, so no bound. Listed explicitly (no `_`) so a new
        // `AExpr` variant fails to compile here and forces a decision.
        AExpr::Element
        | AExpr::Explode { .. }
        | AExpr::Column(_)
        | AExpr::Literal(_)
        | AExpr::Cast { .. }
        | AExpr::Sort { .. }
        | AExpr::Gather { .. }
        | AExpr::SortBy { .. }
        | AExpr::Filter { .. }
        | AExpr::Agg(_)
        | AExpr::Ternary { .. }
        | AExpr::AnonymousAgg { .. }
        | AExpr::AnonymousFunction { .. }
        | AExpr::Eval { .. }
        | AExpr::Over { .. }
        | AExpr::Slice { .. }
        | AExpr::Len => Classification::Opaque,
        #[cfg(feature = "dtype-struct")]
        AExpr::StructField(_) => Classification::Opaque,
        #[cfg(feature = "dtype-struct")]
        AExpr::StructEval { .. } => Classification::Opaque,
        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling { .. } => Classification::Opaque,
    }
}

// Classifies `col op lit` / `lit op col`, plus `col == col` as an `Equality` edge.
fn classify_comparison(
    left: Node,
    op: Operator,
    right: Node,
    expr_arena: &Arena<AExpr>,
    constraints: &mut PlIndexMap<PlSmallStr, ColumnConstraints>,
) -> Classification {
    let left_ae = expr_arena.get(left);
    let right_ae = expr_arena.get(right);

    // Normalize to `col op lit`; both-columns or both-literals has no single-column bound.
    let (col_name, lit, op) =
        if let (Some(name), Some(lit)) = (as_column(left_ae), as_scalar_lit(right_ae)) {
            (name, lit, op)
        } else if let (Some(lit), Some(name)) = (as_scalar_lit(left_ae), as_column(right_ae)) {
            (name, lit, flip_comparison(op))
        } else if let (Some(a), Some(b)) = (as_column(left_ae), as_column(right_ae)) {
            // Both sides columns. Any ordinary comparison drops rows where either
            // operand is null (K3), so both columns are non-null on surviving rows;
            // recording that lets e.g. `a <= b AND a.is_null()` collapse. The null-safe
            // forms (`EqValidity` / `NotEqValidity`) keep nulls, so they're excluded.
            // `==` additionally yields a cross-column propagation edge (both ends share
            // a value on every surviving row); the others are kept verbatim.
            if matches!(
                op,
                Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
            ) {
                constraints
                    .entry(a.clone())
                    .or_default()
                    .require_nullability(Nullability::NonNull);
                constraints
                    .entry(b.clone())
                    .or_default()
                    .require_nullability(Nullability::NonNull);
                if op == Operator::Eq {
                    return Classification::Equality(a, b);
                }
            }
            return Classification::Opaque;
        } else {
            return Classification::Opaque;
        };

    let cc = constraints.entry(col_name).or_default();
    match op {
        Operator::Gt => {
            cc.add_lower(lit, false);
        },
        Operator::GtEq => {
            cc.add_lower(lit, true);
        },
        Operator::Lt => {
            cc.add_upper(lit, false);
        },
        Operator::LtEq => {
            cc.add_upper(lit, true);
        },
        // `== x` is an inclusive lower and upper bound, both `x`.
        Operator::Eq => {
            cc.add_lower(lit.clone(), true);
            cc.add_upper(lit, true);
        },
        // `!= x` is an excluded value, resolved against the bounds in `resolve_deferred`.
        Operator::NotEq => {
            cc.add_excluded(lit);
        },
        // Null-aware comparisons (`eq_missing` / `ne_missing`) keep nulls, so they
        // don't map to a value constraint. Must stay `Opaque`: modeling `ne_missing`
        // as an exclusion would break the `is_null`-vs-value contradiction in
        // `resolve_deferred` (see the invariant noted there).
        Operator::EqValidity | Operator::NotEqValidity => return Classification::Opaque,
        // And/or and arithmetic aren't a `col op lit` comparison.
        Operator::And
        | Operator::Or
        | Operator::Xor
        | Operator::LogicalAnd
        | Operator::LogicalOr
        | Operator::Plus
        | Operator::Minus
        | Operator::Multiply
        | Operator::RustDivide
        | Operator::TrueDivide
        | Operator::FloorDivide
        | Operator::Modulus => return Classification::Opaque,
    }
    if cc.unsat {
        Classification::Unsat
    } else {
        Classification::Bound
    }
}

// Pushes a logical `NOT` inward, the only negations we model:
//   `!(x > 5)` to `x <= 5` (and the other comparisons),
//   `!x.is_in([3, 5, 6])` records 3, 5, 6 as disallowed (node kept verbatim),
//   `!x.is_null()` / `!x.is_not_null()` to the flipped nullability requirement.
// Sound under K3 nulls (both sides drop nulls) and polars' total-order NaN
// handling (NaN is the greatest value, so the complementary comparison agrees on
// every non-null value, NaN included; this would be unsound under IEEE NaN). De
// Morgan over `AND`/`OR` is out of scope (this rule is conjunction-only), so any
// other inner stays opaque.
fn classify_negation(
    inner: Node,
    expr_arena: &Arena<AExpr>,
    constraints: &mut PlIndexMap<PlSmallStr, ColumnConstraints>,
) -> Classification {
    match expr_arena.get(inner) {
        AExpr::BinaryExpr { left, op, right } => {
            let Some(negated) = negate_comparison(*op) else {
                return Classification::Opaque;
            };
            // A folded bound becomes `Normalized` (forces the rewrite that drops the
            // `!` node); `Unsat` / `Equality` (`!(a != b)` is `a == b`) / `Opaque`
            // (`!(a == b)`, two columns) pass through unchanged.
            match classify_comparison(*left, negated, *right, expr_arena, constraints) {
                Classification::Bound => Classification::Normalized,
                other => other,
            }
        },
        // `!is_in(haystack)` disallows each haystack member: recorded for
        // contradiction detection only, the node kept verbatim (never rewritten
        // into `!=` terms). Same conservatism as positive `is_in` (literal,
        // null-free haystack only, enforced by `as_value_set`).
        #[cfg(feature = "is_in")]
        AExpr::Function {
            input,
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { .. }),
            ..
        } => {
            if let (Some(col_name), Some(values)) = (
                as_column(expr_arena.get(input[0].node())),
                as_value_set(expr_arena.get(input[1].node())),
            ) {
                constraints
                    .entry(col_name)
                    .or_default()
                    .add_disallowed(values);
            }
            Classification::Opaque
        },
        // `!(is_null)` is `is_not_null` and vice versa; record the flipped
        // nullability (node kept verbatim, like the direct `is_null` case).
        AExpr::Function {
            input,
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsNull),
            ..
        } => classify_null(
            input[0].node(),
            Nullability::NonNull,
            expr_arena,
            constraints,
        ),
        AExpr::Function {
            input,
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull),
            ..
        } => classify_null(input[0].node(), Nullability::Null, expr_arena, constraints),
        // Negation of any other shape isn't a constraint we model; kept verbatim.
        AExpr::Function { .. }
        | AExpr::Element
        | AExpr::Explode { .. }
        | AExpr::Column(_)
        | AExpr::Literal(_)
        | AExpr::Cast { .. }
        | AExpr::Sort { .. }
        | AExpr::Gather { .. }
        | AExpr::SortBy { .. }
        | AExpr::Filter { .. }
        | AExpr::Agg(_)
        | AExpr::Ternary { .. }
        | AExpr::AnonymousAgg { .. }
        | AExpr::AnonymousFunction { .. }
        | AExpr::Eval { .. }
        | AExpr::Over { .. }
        | AExpr::Slice { .. }
        | AExpr::Len => Classification::Opaque,
        #[cfg(feature = "dtype-struct")]
        AExpr::StructField(_) => Classification::Opaque,
        #[cfg(feature = "dtype-struct")]
        AExpr::StructEval { .. } => Classification::Opaque,
        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling { .. } => Classification::Opaque,
    }
}

// The complementary comparison for negating `!(col op lit)`. `None` for operators
// whose negation we don't model (null-aware, boolean, arithmetic).
fn negate_comparison(op: Operator) -> Option<Operator> {
    match op {
        Operator::Gt => Some(Operator::LtEq),
        Operator::GtEq => Some(Operator::Lt),
        Operator::Lt => Some(Operator::GtEq),
        Operator::LtEq => Some(Operator::Gt),
        Operator::Eq => Some(Operator::NotEq),
        Operator::NotEq => Some(Operator::Eq),
        Operator::EqValidity
        | Operator::NotEqValidity
        | Operator::And
        | Operator::Or
        | Operator::Xor
        | Operator::LogicalAnd
        | Operator::LogicalOr
        | Operator::Plus
        | Operator::Minus
        | Operator::Multiply
        | Operator::RustDivide
        | Operator::TrueDivide
        | Operator::FloorDivide
        | Operator::Modulus => None,
    }
}

// Records an `is_null` (`Null`) / `is_not_null` (`NonNull`) requirement on a bare
// column, for contradiction detection only; the node is kept verbatim. A non-column
// argument (e.g. `(a + b).is_null()`) stays opaque.
fn classify_null(
    col: Node,
    required: Nullability,
    expr_arena: &Arena<AExpr>,
    constraints: &mut PlIndexMap<PlSmallStr, ColumnConstraints>,
) -> Classification {
    let Some(col_name) = as_column(expr_arena.get(col)) else {
        return Classification::Opaque;
    };
    let cc = constraints.entry(col_name).or_default();
    cc.require_nullability(required);
    if cc.unsat {
        Classification::Unsat
    } else {
        Classification::Opaque
    }
}

// Splits `is_between(col, lo, hi, closed)` into a lower and an upper bound.
#[cfg(feature = "is_between")]
fn classify_is_between(
    col: Node,
    lo: Node,
    hi: Node,
    closed: ClosedInterval,
    expr_arena: &Arena<AExpr>,
    constraints: &mut PlIndexMap<PlSmallStr, ColumnConstraints>,
) -> Classification {
    let Some(col_name) = as_column(expr_arena.get(col)) else {
        return Classification::Opaque;
    };
    let Some(lo_lit) = as_scalar_lit(expr_arena.get(lo)) else {
        return Classification::Opaque;
    };
    let Some(hi_lit) = as_scalar_lit(expr_arena.get(hi)) else {
        return Classification::Opaque;
    };
    let (lo_inclusive, hi_inclusive) = match closed {
        ClosedInterval::Both => (true, true),
        ClosedInterval::Left => (true, false),
        ClosedInterval::Right => (false, true),
        ClosedInterval::None => (false, false),
    };
    let cc = constraints.entry(col_name).or_default();
    cc.add_lower(lo_lit, lo_inclusive);
    cc.add_upper(hi_lit, hi_inclusive);
    if cc.unsat {
        Classification::Unsat
    } else {
        Classification::Bound
    }
}

fn as_column(ae: &AExpr) -> Option<PlSmallStr> {
    if let AExpr::Column(name) = ae {
        Some(name.clone())
    } else {
        None
    }
}

// Returns `None` for a null literal, so we only reason about real values.
fn as_scalar_lit(ae: &AExpr) -> Option<Scalar> {
    if let AExpr::Literal(LiteralValue::Scalar(s)) = ae {
        if matches!(s.value(), AnyValue::Null) {
            return None;
        }
        Some(s.clone())
    } else {
        None
    }
}

// Larger `is_in` haystacks aren't worth modelling (intersecting them costs more
// than the rare contradiction). 100 matches the `is_in` haystack cap used for
// parquet row-group pruning (`skip_batches::LIST_ITEM_LIMIT`).
#[cfg(feature = "is_in")]
const MAX_IS_IN_VALUES: usize = 100;

// The inner value-series of a `List` / `Array` value, else `None`.
#[cfg(feature = "is_in")]
fn list_inner(av: &AnyValue) -> Option<Series> {
    match av {
        AnyValue::List(inner) => Some(inner.clone()),
        #[cfg(feature = "dtype-array")]
        AnyValue::Array(inner, _) => Some(inner.clone()),
        _ => None,
    }
}

// Extracts an `is_in` haystack as scalars. The haystack is one list of values,
// wrapped either as a `Scalar` holding a `List` / `Array` (DSL `is_in([..])`) or a
// length-1 `Series` of that one list (SQL `IN (..)`); both unwrap to the inner
// series whose elements are the values. Bails on other shapes, an oversized
// haystack, or a null member.
#[cfg(feature = "is_in")]
fn as_value_set(ae: &AExpr) -> Option<Vec<Scalar>> {
    let AExpr::Literal(lit) = ae else {
        return None;
    };
    let values: Series = match lit {
        LiteralValue::Scalar(s) => list_inner(s.value())?,
        LiteralValue::Series(s) if s.len() == 1 => list_inner(&s.get(0).ok()?)?,
        _ => return None,
    };

    if values.len() > MAX_IS_IN_VALUES {
        return None;
    }

    let dtype = values.dtype();
    let mut scalars = Vec::with_capacity(values.len());
    for av in values.iter() {
        if matches!(av, AnyValue::Null) {
            return None;
        }
        scalars.push(Scalar::new(dtype.clone(), av.into_static()));
    }
    Some(scalars)
}

// Collects the tightest comparisons for one column as borrowed `(name, op, value)`.
// Kept arena- and clone-free so the caller can count them and bail before cloning
// or adding any node on the common no-op path.
fn collect_column_comparisons<'a>(
    name: &'a PlSmallStr,
    cc: &'a ColumnConstraints,
    out: &mut Vec<(&'a PlSmallStr, Operator, &'a Scalar)>,
) {
    // A pinned column is just `col == value` (no exclusion survives a point range).
    if let Some(value) = cc.fixed_value() {
        out.push((name, Operator::Eq, value));
        return;
    }
    if let Some((lo, inclusive)) = &cc.lower {
        let op = if *inclusive {
            Operator::GtEq
        } else {
            Operator::Gt
        };
        out.push((name, op, lo));
    }
    if let Some((hi, inclusive)) = &cc.upper {
        let op = if *inclusive {
            Operator::LtEq
        } else {
            Operator::Lt
        };
        out.push((name, op, hi));
    }
    for value in &cc.excluded {
        out.push((name, Operator::NotEq, value));
    }
}

// Builds a `col op value` node.
fn comparison_node(
    name: &PlSmallStr,
    op: Operator,
    value: Scalar,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    let left = expr_arena.add(AExpr::Column(name.clone()));
    let right = expr_arena.add(AExpr::Literal(value.into()));
    expr_arena.add(AExpr::BinaryExpr { left, op, right })
}

// Folds the conjuncts into a left-deep `AND` chain. Every rewrite trigger keeps an
// opaque node or emits a comparison, so there is always at least one conjunct.
fn fold_and(nodes: Vec<Node>, expr_arena: &mut Arena<AExpr>) -> Node {
    let mut nodes = nodes.into_iter();
    let mut acc = nodes.next().expect("at least one conjunct");
    for node in nodes {
        acc = expr_arena.add(AExpr::BinaryExpr {
            left: acc,
            op: Operator::And,
            right: node,
        });
    }
    acc
}

fn flip_comparison(op: Operator) -> Operator {
    match op {
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Eq => Operator::Eq,
        Operator::NotEq => Operator::NotEq,
        // Other operators have no flipped form; return them unchanged. They
        // never reach a real comparison, so the value doesn't matter.
        Operator::EqValidity
        | Operator::NotEqValidity
        | Operator::And
        | Operator::Or
        | Operator::Xor
        | Operator::LogicalAnd
        | Operator::LogicalOr
        | Operator::Plus
        | Operator::Minus
        | Operator::Multiply
        | Operator::RustDivide
        | Operator::TrueDivide
        | Operator::FloorDivide
        | Operator::Modulus => op,
    }
}
