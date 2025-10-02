use polars_utils::arena::Arena;

use crate::plans::AExpr;
use crate::plans::set_order::expr_pushdown::{
    ExprOutputOrder, FrameOrderObserved, ObservableOrderingsResolver,
};

pub fn is_output_ordered(aexpr: &AExpr, arena: &Arena<AExpr>, frame_ordered: bool) -> bool {
    use ExprOutputOrder as O;

    match ObservableOrderingsResolver::new(if frame_ordered {
        O::Independent
    } else {
        O::None
    })
    .resolve_observable_orderings(aexpr, arena)
    {
        Ok(O::None) => false,
        Ok(O::Independent) => true,

        // Sorts always output in a defined ordering.
        AExpr::Sort { .. } | AExpr::SortBy { .. } => true,
        AExpr::Gather {
            expr: _,
            idx,
            returns_scalar,
        } => !returns_scalar && rec!(*idx),

        // Filter propagates the input order.
        AExpr::Filter { input, by: _ } => rec!(*input),

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
    }
}
