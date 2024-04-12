use polars_ops::prelude::*;
use polars_plan::logical_plan::expr_ir::ExprIR;
use polars_plan::prelude::*;

pub(super) fn is_streamable_sort(args: &SortArguments) -> bool {
    // check if slice is positive or maintain order is true
    match args {
        SortArguments {
            maintain_order: true,
            ..
        } => false,
        SortArguments {
            slice: Some((offset, _)),
            ..
        } => *offset >= 0,
        SortArguments { slice: None, .. } => true,
    }
}

/// check if all expressions are a simple column projection
pub(super) fn all_column(exprs: &[ExprIR], expr_arena: &Arena<AExpr>) -> bool {
    exprs
        .iter()
        .all(|e| matches!(expr_arena.get(e.node()), AExpr::Column(_)))
}

pub(super) fn streamable_join(args: &JoinArgs) -> bool {
    let supported = match args.how {
        #[cfg(feature = "cross_join")]
        JoinType::Cross => true,
        JoinType::Inner | JoinType::Left | JoinType::Outer { .. } => true,
        _ => false,
    };
    supported && !args.validation.needs_checks()
}
