use super::*;

fn has_series_or_range(ae: &AExpr) -> bool {
    matches!(
        ae,
        AExpr::Literal(LiteralValue::Series(_) | LiteralValue::Range { .. })
    )
}

pub fn is_streamable(node: Node, expr_arena: &Arena<AExpr>, context: Context) -> bool {
    // check whether leaf column is Col or Lit
    let mut seen_column = false;
    let mut seen_lit_range = false;
    let all = expr_arena.iter(node).all(|(_, ae)| match ae {
        AExpr::Function {
            function: FunctionExpr::SetSortedFlag(_),
            ..
        } => true,
        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => match context
        {
            Context::Default => matches!(
                options.collect_groups,
                ApplyOptions::ElementWise | ApplyOptions::ApplyList
            ),
            Context::Aggregation => matches!(options.collect_groups, ApplyOptions::ElementWise),
        },
        AExpr::Column(_) => {
            seen_column = true;
            true
        },
        AExpr::BinaryExpr { left, right, .. } => {
            !has_aexpr(*left, expr_arena, has_series_or_range)
                && !has_aexpr(*right, expr_arena, has_series_or_range)
        },
        AExpr::Ternary {
            truthy,
            falsy,
            predicate,
        } => {
            !has_aexpr(*truthy, expr_arena, has_series_or_range)
                && !has_aexpr(*falsy, expr_arena, has_series_or_range)
                && !has_aexpr(*predicate, expr_arena, has_series_or_range)
        },
        AExpr::Alias(_, _) | AExpr::Cast { .. } => true,
        AExpr::Literal(lv) => match lv {
            LiteralValue::Series(_) | LiteralValue::Range { .. } => {
                seen_lit_range = true;
                true
            },
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

pub fn all_streamable(exprs: &[ExprIR], expr_arena: &Arena<AExpr>, context: Context) -> bool {
    exprs
        .iter()
        .all(|e| is_streamable(e.node(), expr_arena, context))
}
