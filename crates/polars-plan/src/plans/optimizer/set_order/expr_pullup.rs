use polars_utils::arena::Arena;

use crate::dsl::EvalVariant;
use crate::plans::{AExpr, IRAggExpr};

/// Determine whether the output of an expression has a defined order.
///
/// This will recursively walk through the expression and answers the question:
///
/// > Given that the input dataframe (does not have/has) a defined ordered, does the expression
/// > have a defined output order?
#[recursive::recursive]
pub fn is_output_ordered(aexpr: &AExpr, arena: &Arena<AExpr>, frame_ordered: bool) -> bool {
    macro_rules! rec {
        ($node:expr) => {{ is_output_ordered(arena.get($node), arena, frame_ordered) }};
    }
    match aexpr {
        // Explode creates local orders.
        AExpr::Explode { .. } => true,
        AExpr::Column(..) => frame_ordered,
        AExpr::Literal(lv) => !lv.is_scalar(),

        // All elementwise expressions are ordered if any of its inputs are ordered.
        AExpr::BinaryExpr { left, right, op: _ } => rec!(*left) || rec!(*right),
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => rec!(*predicate) || rec!(*truthy) || rec!(*falsy),
        AExpr::Cast {
            expr,
            dtype: _,
            options: _,
        } => rec!(*expr),

        // Sorts always output in a defined ordering.
        AExpr::Sort { .. } | AExpr::SortBy { .. } => true,
        AExpr::Gather {
            expr: _,
            idx,
            returns_scalar,
        } => !returns_scalar && rec!(*idx),
        AExpr::Filter { input, by } => rec!(*input) || rec!(*by),

        // This aggregation is jiberish. Just be conservative.
        AExpr::Agg(IRAggExpr::AggGroups(_)) => true,

        // Aggregations always output 1 row.
        AExpr::Agg(..) | AExpr::Len => false,

        AExpr::Eval {
            expr,
            evaluation: _,
            variant,
        } => match variant {
            EvalVariant::List => rec!(*expr),
            EvalVariant::Cumulative { min_samples: _ } => true,
        },
        AExpr::AnonymousFunction { input, options, .. }
        | AExpr::Function { input, options, .. } => {
            if options.flags.returns_scalar() || options.flags.is_output_unordered() {
                false
            } else if options.is_elementwise() {
                input.iter().any(|e| rec!(e.node()))
            } else if options.flags.propagates_order() {
                assert_eq!(input.len(), 1);
                rec!(input[0].node())
            } else {
                true
            }
        },
        // @Performance. This is probably quite pessimistic and can be optimizes to be `false` in
        // some cases.
        AExpr::Window { .. } => true,
        AExpr::Slice {
            input,
            offset: _,
            length: _,
        } => rec!(*input),
    }
}
