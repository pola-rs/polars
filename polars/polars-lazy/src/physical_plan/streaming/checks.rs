use polars_core::prelude::{JoinArgs, JoinType};
use polars_plan::prelude::*;

pub(super) fn is_streamable_sort(args: &SortArguments) -> bool {
    // check if slice is positive
    match args.slice {
        Some((offset, _)) => offset >= 0,
        None => true,
    }
}

pub(super) fn is_streamable(node: Node, expr_arena: &Arena<AExpr>, context: Context) -> bool {
    // check whether leaf column is Col or Lit
    let mut seen_column = false;
    let mut seen_lit_range = false;
    let all = expr_arena.iter(node).all(|(_, ae)| match ae {
        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => match context
        {
            Context::Default => matches!(
                options.collect_groups,
                ApplyOptions::ApplyFlat | ApplyOptions::ApplyList
            ),
            Context::Aggregation => matches!(options.collect_groups, ApplyOptions::ApplyFlat),
        },
        AExpr::Column(_) => {
            seen_column = true;
            true
        }
        AExpr::Ternary { .. }
        | AExpr::BinaryExpr { .. }
        | AExpr::Alias(_, _)
        | AExpr::Cast { .. } => true,
        AExpr::Literal(lv) => match lv {
            LiteralValue::Series(_) | LiteralValue::Range { .. } => {
                seen_lit_range = true;
                true
            }
            _ => true,
        },
        _ => false,
    });

    if all {
        // adding a range or literal series to chunks will fail because sizes don't match
        // if column is a leaf column then it is ok
        // - so we want to block `with_column(lit(Series))`
        // - but we want to allow `with_column(col("foo").is_in(Series))`
        // that means that IFF we seen a lit_range, we only allow if we also seen a `column`.
        return if seen_lit_range { seen_column } else { true };
    }
    false
}

pub(super) fn all_streamable(exprs: &[Node], expr_arena: &Arena<AExpr>, context: Context) -> bool {
    exprs
        .iter()
        .all(|node| is_streamable(*node, expr_arena, context))
}

/// check if all expressions are a simple column projection
pub(super) fn all_column(exprs: &[Node], expr_arena: &Arena<AExpr>) -> bool {
    exprs
        .iter()
        .all(|node| matches!(expr_arena.get(*node), AExpr::Column(_)))
}

pub(super) fn streamable_join(args: &JoinArgs) -> bool {
    let supported = match args.how {
        #[cfg(feature = "cross_join")]
        JoinType::Cross => true,
        JoinType::Inner | JoinType::Left => true,
        _ => false,
    };
    supported && !args.validation.needs_checks()
}
