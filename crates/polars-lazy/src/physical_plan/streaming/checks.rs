use polars_core::chunked_array::ops::SortMultipleOptions;
use polars_ops::prelude::*;
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::prelude::*;

pub(super) fn is_streamable_sort(
    slice: &Option<(i64, usize)>,
    sort_options: &SortMultipleOptions,
) -> bool {
    // check if slice is positive or maintain order is true
    if sort_options.maintain_order {
        false
    } else if let Some((offset, _)) = slice {
        *offset >= 0
    } else {
        true
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
        JoinType::Left => true,
        JoinType::Inner => {
            // no-coalescing not yet supported in streaming
            matches!(
                args.coalesce,
                JoinCoalesce::JoinSpecific | JoinCoalesce::CoalesceColumns
            )
        },
        JoinType::Full { .. } => true,
        _ => false,
    };
    supported && !args.validation.needs_checks()
}
