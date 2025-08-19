//! Order sensitivity is a property of an operation that says that if the inputs' rows are
//! reordered before the operation takes place, the operation returns different results even when
//! not regarding order.
//!
//! Operations that are not order sensitive are a superset of *row separable* operations.
//! Usually operations that are not row separable and not order sensitive (e.g. aggregations) need
//! to see the whole data before being about to provide the correct output.
//!
//! Below is a formal definition.
//!
//! A shuffle of a series A with seed S is given as:
//!
//! Shuffle(A, S) =
//!     CONCAT(A[P[i]] for i in [0, |A|))
//!     where
//!         P is a seeded by S random permutation of all integers in [0, |A|)
//!
//! An operation f(A1, ..., An) is order sensitive i.f.f.
//!
//! sort(f(A1, ..., An)) != sort(f(Shuffle(A1, S), ..., Shuffle(An, S))) for any seed S

use super::super::*;
use crate::plans::IRAggExpr;

pub fn is_order_sensitive(aexpr: &AExpr, arena: &Arena<AExpr>) -> bool {
    is_order_sensitive_amortized(aexpr, arena, &mut Vec::new())
}

pub fn is_order_sensitive_amortized(
    aexpr: &AExpr,
    arena: &Arena<AExpr>,
    stack: &mut Vec<Node>,
) -> bool {
    if is_order_sensitive_top_level(aexpr) {
        return true;
    }

    stack.clear();
    aexpr.inputs_rev(stack);

    while let Some(node) = stack.pop() {
        let aexpr = arena.get(node);
        if is_order_sensitive_top_level(aexpr) {
            return true;
        }
        aexpr.inputs_rev(stack);
    }

    false
}

pub fn is_order_sensitive_top_level(aexpr: &AExpr) -> bool {
    match aexpr {
        AExpr::Explode {
            expr: _,
            skip_empty: _,
        } => false,
        AExpr::Column(_) => false,
        AExpr::Literal(_) => false,
        AExpr::BinaryExpr {
            left: _,
            op: _,
            right: _,
        } => false,
        AExpr::Cast {
            expr: _,
            dtype: _,
            options: _,
        } => false,
        AExpr::Sort { expr: _, options } => !options.maintain_order,
        AExpr::Gather {
            expr: _,
            idx: _,
            returns_scalar: _,
        } => false,
        AExpr::SortBy {
            expr: _,
            by: _,
            sort_options,
        } => !sort_options.maintain_order,
        AExpr::Filter { input: _, by: _ } => false,
        AExpr::Agg(agg) => is_order_sensitive_agg_top_level(agg),
        AExpr::Ternary {
            predicate: _,
            truthy: _,
            falsy: _,
        } => false,

        AExpr::AnonymousFunction {
            input: _,
            function: _,
            options,
            fmt_str: _,
        }
        | AExpr::Function {
            input: _,
            function: _,
            options,
        } => !options.is_row_separable(),
        AExpr::Eval {
            expr: _,
            evaluation: _,
            variant,
        } => !variant.is_row_separable(),
        AExpr::Window {
            function: _,
            partition_by: _,
            order_by: _,
            options: _,
        } => true,
        AExpr::Slice {
            input: _,
            offset: _,
            length: _,
        } => true,
        AExpr::Len => false,
    }
}

pub fn is_order_sensitive_agg_top_level(agg: &IRAggExpr) -> bool {
    match agg {
        IRAggExpr::Min {
            input: _,
            propagate_nans: _,
        } => false,
        IRAggExpr::Max {
            input: _,
            propagate_nans: _,
        } => false,
        IRAggExpr::Median(_) => false,
        IRAggExpr::NUnique(_) => false,
        IRAggExpr::First(_) => true,
        IRAggExpr::Last(_) => true,
        IRAggExpr::Mean(_) => false,
        IRAggExpr::Implode(_) => true,
        IRAggExpr::Quantile {
            expr: _,
            quantile: _,
            method: _,
        } => false,
        IRAggExpr::Sum(_) => false,
        IRAggExpr::Count {
            input: _,
            include_nulls: _,
        } => false,
        IRAggExpr::Std(_, _) => false,
        IRAggExpr::Var(_, _) => false,
        IRAggExpr::AggGroups(_) => true,
    }
}
