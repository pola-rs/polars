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
            IRAggExpr::Implode(_) => false,
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
        if !is_order_dependent_top_level(ae, ctx) {
            return false;
        }

        let Some(node) = stack.pop() else {
            break;
        };

        ae = expr_arena.get(node);
    }

    true
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
    nodes
        .iter()
        .all(|n| !is_order_dependent(expr_arena.get(n.into()), expr_arena, ctx))
}

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
            IR::Sort { .. } => {
                maintain_order_above = false;
            },
            IR::Distinct { options, .. } => {
                if !maintain_order_above {
                    options.maintain_order = false;
                    continue;
                }
                if !options.maintain_order {
                    maintain_order_above = false;
                }
            },
            IR::Union { options, .. } => {
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
                if !maintain_order_above && *maintain_order {
                    *maintain_order = false;
                    continue;
                }

                if apply.is_some()
                    || *maintain_order
                    || options.is_rolling()
                    || options.is_dynamic()
                {
                    maintain_order_above = true;
                    continue;
                }
                if all_elementwise(keys, expr_arena)
                    && !all_order_independent(aggs, expr_arena, Context::Aggregation)
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
                maintain_order_above = true;
            },
        }
    }
}
