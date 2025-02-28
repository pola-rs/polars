use polars_utils::unitvec;

use super::*;

// Can give false positives.
fn is_order_dependent_top_level(ae: &AExpr, ctx: Context) -> bool {
    match ae {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Min { .. } => false,
            IRAggExpr::Max { .. } => false,
            IRAggExpr::Median(_) => false,
            IRAggExpr::NUnique(_) => false,
            IRAggExpr::First(_) => true,
            IRAggExpr::Last(_) => true,
            IRAggExpr::Mean(_) => false,
            IRAggExpr::Implode(_) => true,
            IRAggExpr::Quantile { .. } => false,
            IRAggExpr::Sum(_) => false,
            IRAggExpr::Count(_, _) => false,
            IRAggExpr::Std(_, _) => false,
            IRAggExpr::Var(_, _) => false,
            IRAggExpr::AggGroups(_) => true,
        },
        AExpr::Column(_) => matches!(ctx, Context::Aggregation),
        _ => true,
    }
}

// Can give false positives.
fn is_order_dependent<'a>(mut ae: &'a AExpr, expr_arena: &'a Arena<AExpr>, ctx: Context) -> bool {
    let mut stack = unitvec![];

    loop {
        if is_order_dependent_top_level(ae, ctx) {
            return true;
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    false
}

// Can give false negatives.
pub(crate) fn all_order_independent<'a, N>(
    nodes: &'a [N],
    expr_arena: &Arena<AExpr>,
    ctx: Context,
) -> bool
where
    Node: From<&'a N>,
{
    !nodes
        .iter()
        .any(|n| is_order_dependent(expr_arena.get(n.into()), expr_arena, ctx))
}

// Should run before slice pushdown.
pub(super) fn set_order_flags(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    scratch: &mut Vec<Node>,
) {
    scratch.clear();
    scratch.push(root);

    let mut maintain_order_above = true;

    while let Some(node) = scratch.pop() {
        let ir = ir_arena.get_mut(node);
        ir.copy_inputs(scratch);

        match ir {
            IR::Sort {
                input,
                sort_options,
                ..
            } => {
                debug_assert!(sort_options.limit.is_none());
                // This sort can be removed
                if !maintain_order_above {
                    scratch.pop();
                    scratch.push(node);
                    let input = *input;
                    ir_arena.swap(node, input);
                    continue;
                }

                if !sort_options.maintain_order {
                    maintain_order_above = false; // `maintain_order=True` is influenced by result of earlier sorts
                }
            },
            IR::Distinct { options, .. } => {
                debug_assert!(options.slice.is_none());
                if !maintain_order_above {
                    options.maintain_order = false;
                    continue;
                }
                if matches!(
                    options.keep_strategy,
                    UniqueKeepStrategy::First | UniqueKeepStrategy::Last
                ) {
                    maintain_order_above = true;
                } else if !options.maintain_order {
                    maintain_order_above = false;
                }
            },
            IR::Union { options, .. } => {
                debug_assert!(options.slice.is_none());
                options.maintain_order = maintain_order_above;
            },
            IR::GroupBy {
                keys,
                aggs,
                maintain_order,
                options,
                apply,
                ..
            } => {
                debug_assert!(options.slice.is_none());
                if apply.is_some()
                    || *maintain_order
                    || options.is_rolling()
                    || options.is_dynamic()
                {
                    maintain_order_above = true;
                    continue;
                }
                if !maintain_order_above && *maintain_order {
                    *maintain_order = false;
                    continue;
                }

                if all_elementwise(keys, expr_arena)
                    && all_order_independent(aggs, expr_arena, Context::Aggregation)
                {
                    maintain_order_above = false;
                    continue;
                }
                maintain_order_above = true;
            },
            // Conservative now.
            IR::HStack { exprs, .. } | IR::Select { expr: exprs, .. } => {
                if !maintain_order_above && all_elementwise(exprs, expr_arena) {
                    continue;
                }
                maintain_order_above = true;
            },
            _ => {
                // If we don't know maintain order
                // Known: slice
                maintain_order_above = true;
            },
        }
    }
}
