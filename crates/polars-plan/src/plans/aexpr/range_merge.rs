//! Combine per-column comparisons in an `AND` chain to spot impossible filters.

use std::cmp::Ordering;

use polars_core::prelude::AnyValue;
use polars_core::scalar::Scalar;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::properties::ExprPushdownGroup;
use super::{AExpr, LiteralValue, MintermIter, Operator};
#[cfg(feature = "is_between")]
use super::{IRBooleanFunction, IRFunctionExpr};
#[cfg(feature = "is_between")]
use crate::prelude::ClosedInterval;

#[derive(Default)]
struct ColumnConstraints {
    // The bool is whether the bound is inclusive: true for `>=`/`<=`, false for `>`/`<`.
    // `== x` is stored as an inclusive lower and upper bound, both `x`.
    lower: Option<(Scalar, bool)>,
    upper: Option<(Scalar, bool)>,
    // Once true, stays true.
    unsat: bool,
}

impl ColumnConstraints {
    // Keep whichever bound is tighter. `tighter` is the ordering of (new vs
    // existing) that means the new bound wins: `Greater` for a lower bound,
    // `Less` for an upper bound.
    fn tighten(
        slot: &mut Option<(Scalar, bool)>,
        value: Scalar,
        inclusive: bool,
        tighter: Ordering,
    ) {
        *slot = Some(match slot.take() {
            None => (value, inclusive),
            Some((existing, existing_inclusive)) => {
                match value.as_any_value().partial_cmp(&existing.as_any_value()) {
                    Some(ord) if ord == tighter => (value, inclusive),
                    Some(Ordering::Equal) => (value, inclusive && existing_inclusive),
                    // Less tight, or incomparable types: keep the existing bound.
                    _ => (existing, existing_inclusive),
                }
            },
        });
    }

    fn add_lower(&mut self, value: Scalar, inclusive: bool) {
        if self.unsat {
            return;
        }
        Self::tighten(&mut self.lower, value, inclusive, Ordering::Greater);
        self.check_consistency();
    }

    fn add_upper(&mut self, value: Scalar, inclusive: bool) {
        if self.unsat {
            return;
        }
        Self::tighten(&mut self.upper, value, inclusive, Ordering::Less);
        self.check_consistency();
    }

    // Marks the column impossible if the lower bound is now above the upper
    // bound (an empty range). Run after each add.
    fn check_consistency(&mut self) {
        if self.unsat {
            return;
        }
        if let (Some((lo, lo_inc)), Some((hi, hi_inc))) = (&self.lower, &self.upper) {
            match lo.as_any_value().partial_cmp(&hi.as_any_value()) {
                Some(Ordering::Greater) => self.unsat = true,
                Some(Ordering::Equal) if !(*lo_inc && *hi_inc) => self.unsat = true,
                _ => {},
            }
        }
    }
}

/// If the comparisons in `predicate`'s `AND` chain can never all be true at
/// once, returns a new `Literal(false)` node. Otherwise returns `None`.
pub(crate) fn merge_ranges_in_predicate(
    predicate: Node,
    maintain_errors: bool,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    // Walk each part of the AND chain and collect the bounds per column.
    // Shapes we don't understand are simply ignored. Once one column is
    // impossible the whole filter is false, so we can stop early.
    let mut constraints: PlHashMap<PlSmallStr, ColumnConstraints> = PlHashMap::new();
    let mut unsat = false;

    for conjunct in MintermIter::new(predicate, expr_arena) {
        let ae = expr_arena.get(conjunct);
        if classify_into_constraints(ae, expr_arena, &mut constraints) {
            unsat = true;
            break;
        }
    }

    if !unsat {
        return None;
    }

    // The filter is impossible. Replacing it with `false` lets the filter
    // collapse to an empty scan, so `predicate` is never evaluated. That means
    // an expression that would have errored no longer runs, so we only do this
    // when the user has not asked to keep errors (same rule as predicate
    // pushdown). We don't need a separate check for random expressions: only
    // `col op lit` parts decide the contradiction, and a plain column is always
    // deterministic. Checked last so the common (satisfiable) case skips it.
    let mut group = ExprPushdownGroup::Pushable;
    group.update_with_expr_rec(expr_arena.get(predicate), expr_arena, None);
    if group.blocks_pushdown(maintain_errors) {
        return None;
    }

    Some(expr_arena.add(AExpr::Literal(Scalar::from(false).into())))
}

// Records one comparison under its column and returns `true` if that made the
// column impossible. Shapes we don't recognize add nothing and return `false`.
fn classify_into_constraints(
    ae: &AExpr,
    expr_arena: &Arena<AExpr>,
    constraints: &mut PlHashMap<PlSmallStr, ColumnConstraints>,
) -> bool {
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
            #[cfg(not(feature = "is_between"))]
            {
                let _ = (input, function);
            }
            false
        },
        // These shapes aren't a `col op lit` comparison, so they add no bound.
        // Listed one by one (no `_`) so that adding a new `AExpr` variant fails
        // to compile here and forces a decision.
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
        | AExpr::Len => false,
        #[cfg(feature = "dtype-struct")]
        AExpr::StructField(_) => false,
        #[cfg(feature = "dtype-struct")]
        AExpr::StructEval { .. } => false,
        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling { .. } => false,
    }
}

// Handles both `col op lit` and `lit op col`. Returns `true` if the new bound
// made the column impossible.
fn classify_comparison(
    left: Node,
    op: Operator,
    right: Node,
    expr_arena: &Arena<AExpr>,
    constraints: &mut PlHashMap<PlSmallStr, ColumnConstraints>,
) -> bool {
    let left_ae = expr_arena.get(left);
    let right_ae = expr_arena.get(right);

    // Put it in `col op lit` form. If both sides are columns or both are
    // literals, there's no single-column bound to record.
    let (col_name, lit, op) =
        if let (Some(name), Some(lit)) = (as_column(left_ae), as_scalar_lit(right_ae)) {
            (name, lit, op)
        } else if let (Some(lit), Some(name)) = (as_scalar_lit(left_ae), as_column(right_ae)) {
            (name, lit, flip_comparison(op))
        } else {
            return false;
        };

    let cc = constraints.entry(col_name).or_default();
    match op {
        Operator::Gt => cc.add_lower(lit, false),
        Operator::GtEq => cc.add_lower(lit, true),
        Operator::Lt => cc.add_upper(lit, false),
        Operator::LtEq => cc.add_upper(lit, true),
        // `== x` is an inclusive lower and upper bound, both `x`.
        Operator::Eq => {
            cc.add_lower(lit.clone(), true);
            cc.add_upper(lit, true);
        },
        // `!=` would need set math; not handled.
        Operator::NotEq => return false,
        // Null-aware comparisons don't map to a simple range (nulls behave
        // differently here).
        Operator::EqValidity | Operator::NotEqValidity => return false,
        // And/or and arithmetic ops aren't a `col op lit` comparison, so skip.
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
        | Operator::Modulus => return false,
    }
    cc.unsat
}

// Splits `is_between(col, lo, hi, closed)` into a lower and an upper bound.
// Returns `true` if those bounds made the column impossible.
#[cfg(feature = "is_between")]
fn classify_is_between(
    col: Node,
    lo: Node,
    hi: Node,
    closed: ClosedInterval,
    expr_arena: &Arena<AExpr>,
    constraints: &mut PlHashMap<PlSmallStr, ColumnConstraints>,
) -> bool {
    let Some(col_name) = as_column(expr_arena.get(col)) else {
        return false;
    };
    let Some(lo_lit) = as_scalar_lit(expr_arena.get(lo)) else {
        return false;
    };
    let Some(hi_lit) = as_scalar_lit(expr_arena.get(hi)) else {
        return false;
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
    cc.unsat
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
