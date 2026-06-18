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
//! 3. **Resolve** (`resolve_deferred`): resolve `!=` / `is_in` against the now
//!    final bounds; needs propagated bounds, so runs after stage 2.
//! 4. **Emit**: `false` if impossible (gated on `maintain_errors`), else the
//!    tightest rebuilt chain, or `None` if unchanged.

use std::cmp::Ordering;

use polars_core::prelude::AnyValue;
#[cfg(feature = "is_in")]
use polars_core::prelude::Series;
use polars_core::scalar::Scalar;
use polars_utils::aliases::{InitHashMaps, PlIndexMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::properties::ExprPushdownGroup;
use super::{AExpr, LiteralValue, MintermIter, Operator};
#[cfg(any(feature = "is_between", feature = "is_in"))]
use super::{IRBooleanFunction, IRFunctionExpr};
#[cfg(feature = "is_between")]
use crate::prelude::ClosedInterval;

// How a single conjunct in the `AND` chain was classified.
enum Classification {
    // Bounds crossed (empty range) during the walk, so the filter is impossible.
    // The only emptiness caught here; cross-column conflicts surface in
    // `propagate_equalities`, `!=` / `is_in` in `resolve_deferred`.
    Unsat,
    // A bound folded into its column and rebuilt from the merged constraints,
    // so the original conjunct is dropped.
    Bound,
    // A `col == col` equality, whose two columns form a cross-column propagation
    // edge (the node is kept verbatim unless its class becomes fixed).
    Equality(PlSmallStr, PlSmallStr),
    // Node kept verbatim: either unmodeled (UDFs, null-aware ops, arithmetic) or
    // modeled only for contradiction (`is_in` records its haystack, not rewritten).
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

// Orders two scalars; incomparable types give `None` (neither equal nor ordered),
// so we never declare a contradiction we aren't sure of.
fn scalar_cmp(a: &Scalar, b: &Scalar) -> Option<Ordering> {
    a.as_any_value().partial_cmp(&b.as_any_value())
}

fn scalar_eq(a: &Scalar, b: &Scalar) -> bool {
    scalar_cmp(a, b) == Some(Ordering::Equal)
}

#[derive(Clone, Default)]
struct ColumnConstraints {
    // `bool` = inclusive (`>=`/`<=` vs `>`/`<`). `== x` is an inclusive `[x, x]`.
    lower: Option<(Scalar, bool)>,
    upper: Option<(Scalar, bool)>,
    // `!= x` values, resolved against the bounds in `resolve_deferred`.
    excluded: Vec<Scalar>,
    // Intersection of all `is_in` haystacks; resolved against bounds in `resolve_deferred`.
    allowed: Option<Vec<Scalar>>,
    // Empty value set (unsatisfiable). Set by any stage (crossed bounds, `!=`/`is_in`,
    // cross-column conflict) and sticky, so `add_*` bail early once it is set.
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
        self.check_consistency();
        changed
    }

    fn add_upper(&mut self, value: Scalar, inclusive: bool) -> bool {
        if self.unsat {
            return false;
        }
        let changed = Self::tighten(&mut self.upper, value, inclusive, Ordering::Less);
        self.check_consistency();
        changed
    }

    fn add_excluded(&mut self, value: Scalar) -> bool {
        if self.unsat {
            return false;
        }
        // Dedup so repeated `!= x` collapse to one.
        if self.excluded.iter().any(|e| scalar_eq(e, &value)) {
            return false;
        }
        self.excluded.push(value);
        true
    }

    #[cfg(feature = "is_in")]
    fn add_allowed(&mut self, values: Vec<Scalar>) -> bool {
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
                existing.retain(|e| values.iter().any(|v| scalar_eq(e, v)));
                existing.len() != before
            },
        }
    }

    // Intersects another column's constraints into this one (for `col == col`: on a
    // surviving row the two are equal, so each bound/exclusion/`is_in` holds for both).
    // Returns whether anything changed.
    fn merge_from(&mut self, other: &ColumnConstraints) -> bool {
        if self.unsat {
            return false;
        }
        let mut changed = false;
        if let Some((value, inclusive)) = &other.lower {
            changed |= self.add_lower(value.clone(), *inclusive);
        }
        if let Some((value, inclusive)) = &other.upper {
            changed |= self.add_upper(value.clone(), *inclusive);
        }
        for value in &other.excluded {
            changed |= self.add_excluded(value.clone());
        }
        #[cfg(feature = "is_in")]
        if let Some(values) = &other.allowed {
            changed |= self.add_allowed(values.clone());
        }
        changed
    }

    // Marks the column impossible when the lower bound exceeds the upper (empty range).
    fn check_consistency(&mut self) {
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
    fn resolve_deferred(&mut self) {
        for value in std::mem::take(&mut self.excluded) {
            if self.unsat {
                break;
            }
            match self.classify_excluded(&value) {
                ExcludedResolution::Drop => {},
                ExcludedResolution::Keep => self.excluded.push(value),
                ExcludedResolution::TightenLower => self.lower = Some((value, false)),
                ExcludedResolution::TightenUpper => self.upper = Some((value, false)),
            }
            self.check_consistency();
        }

        // `is_in` confines the column to a finite set; impossible if no member is
        // feasible (disjoint `is_in`, or `is_in([1, 2]) AND a != 1 AND a != 2`).
        if !self.unsat {
            if let Some(allowed) = &self.allowed {
                if !allowed.iter().any(|v| self.is_feasible(v)) {
                    self.unsat = true;
                }
            }
        }
    }

    // Whether `value` survives the bounds and exclusions. Incomparable types count
    // as feasible, so we never declare a contradiction we aren't sure of.
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
        !self.excluded.iter().any(|e| scalar_eq(e, value))
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
    let mut unsat = false;

    for conjunct in MintermIter::new(predicate, expr_arena) {
        match classify_into_constraints(expr_arena.get(conjunct), expr_arena, &mut constraints) {
            Classification::Unsat => {
                unsat = true;
                break;
            },
            Classification::Bound => num_bound_conjuncts += 1,
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
        // as predicate pushdown). No determinism check needed: only `col op lit`
        // parts decide the contradiction, and a plain column is deterministic.
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

    // Rewrite only if we tightened, propagated, or dropped an `a == b`; else leave
    // the plan alone (no churn, no optimizer loop).
    let tightened = comparisons.len() < num_bound_conjuncts;
    if !(tightened || propagated || dropped_equality) {
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
// tightens, so it terminates. Returns whether anything was propagated.
fn propagate_equalities(
    constraints: &mut PlIndexMap<PlSmallStr, ColumnConstraints>,
    edges: &[(PlSmallStr, PlSmallStr, Node)],
) -> bool {
    let mut changed = false;
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
            progressed |= cb.merge_from(ca);
            progressed |= ca.merge_from(cb);
        }
        changed |= progressed;
        if !progressed {
            return changed;
        }
    }
}

// Classifies one conjunct, folding any `col op lit` bound into its column.
// Unmodeled shapes are reported as `Opaque` and kept verbatim by the caller;
// `is_in` is also `Opaque` but still records its haystack for contradiction
// detection.
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
            #[cfg(feature = "is_between")]
            if let IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween { closed }) = function {
                assert_eq!(
                    input.len(),
                    3,
                    "is_between has 3 inputs: value, lower, upper"
                );
                return classify_is_between(
                    input[0].node(),
                    input[1].node(),
                    input[2].node(),
                    *closed,
                    expr_arena,
                    constraints,
                );
            }
            #[cfg(feature = "is_in")]
            if let IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { .. }) = function {
                // Record the haystack as an allowed-set for contradiction detection
                // only; keep the `is_in` node (we don't rewrite it, its null
                // handling is subtle).
                if let Some(col_name) = as_column(expr_arena.get(input[0].node())) {
                    if let Some(values) = as_value_set(expr_arena.get(input[1].node())) {
                        constraints.entry(col_name).or_default().add_allowed(values);
                    }
                }
                return Classification::Opaque;
            }
            #[cfg(not(any(feature = "is_between", feature = "is_in")))]
            {
                let _ = (input, function);
            }
            Classification::Opaque
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
        } else {
            // Both sides columns under `==`: a cross-column equality edge. Create
            // both column entries here (as the bound path does); the driver only
            // records the edge.
            if op == Operator::Eq {
                if let (Some(a), Some(b)) = (as_column(left_ae), as_column(right_ae)) {
                    constraints.entry(a.clone()).or_default();
                    constraints.entry(b.clone()).or_default();
                    return Classification::Equality(a, b);
                }
            }
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
        // Null-aware comparisons don't map to a simple range.
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

// Extracts an `is_in` haystack as scalars (a `Series` literal or a list/array
// scalar). Bails on other shapes, an oversized haystack, or a null member.
#[cfg(feature = "is_in")]
fn as_value_set(ae: &AExpr) -> Option<Vec<Scalar>> {
    let series: &Series = match ae {
        AExpr::Literal(LiteralValue::Series(s)) => s,
        AExpr::Literal(LiteralValue::Scalar(s)) => match s.value() {
            AnyValue::List(inner) => inner,
            #[cfg(feature = "dtype-array")]
            AnyValue::Array(inner, _) => inner,
            _ => return None,
        },
        _ => return None,
    };

    if series.len() > MAX_IS_IN_VALUES {
        return None;
    }

    let dtype = series.dtype();
    let mut values = Vec::with_capacity(series.len());
    for av in series.iter() {
        if matches!(av, AnyValue::Null) {
            return None;
        }
        values.push(Scalar::new(dtype.clone(), av.into_static()));
    }
    Some(values)
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

// Folds the conjuncts into a left-deep `AND` chain. The input is never empty:
// we only rebuild when at least one comparison remains.
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
