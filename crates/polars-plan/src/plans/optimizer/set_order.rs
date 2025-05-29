use polars_utils::unitvec;

use super::*;

// Can give false positives.
fn is_order_dependent_top_level(ae: &AExpr, _ctx: Context) -> bool {
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
        AExpr::Column(_) => false, //For review - does Aggregation Context matter?
        _ => true,
    }
}

// Check if the given node, or, recursively, any of its input nodes in its input
// dependency tree contains an order_dependent operation.
// Return true if an order_dependent node is found.
// Can give false positives.
fn is_order_dependent_rec(node: Node, expr_arena: &Arena<AExpr>, ctx: Context) -> bool {
    let mut stack = unitvec![];
    stack.push(node);

    while let Some(node) = stack.pop() {
        if is_order_dependent_top_level(expr_arena.get(node), ctx) {
            return true;
        }

        let ae = expr_arena.get(node);
        ae.inputs_rev(&mut stack);
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
        .any(|n| is_order_dependent_rec(n.into(), expr_arena, ctx))
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

                if !maintain_order_above {
                    *maintain_order = false;
                }

                if apply.is_some()
                    || *maintain_order
                    || options.is_rolling()
                    || options.is_dynamic()
                {
                    maintain_order_above = true;
                    continue;
                }

                maintain_order_above = !(all_elementwise(keys, expr_arena)
                    && all_order_independent(aggs, expr_arena, Context::Aggregation));
            },
            // Conservative now.
            IR::HStack { exprs, .. } | IR::Select { expr: exprs, .. } => {
                if !maintain_order_above && all_elementwise(exprs, expr_arena) {
                    continue;
                }
                maintain_order_above = true;
            },
            _ => {
                // FIXME:
                // `maintain_order_above` is not correctly propagated in recursion for IR nodes with
                // multiple inputs.
                //
                // This is current not an issue, as we never have an unordered leaf IR node. But
                // if this ends up being the case in the future we need to fix. E.g.:
                //
                // ```
                // q = pl.concat(
                //     [
                //         pl.scan_parquet(..., maintain_order=False), # PLAN 1
                //         pl.LazyFrame(...).sort(...),                # PLAN 2
                //     ]
                // )
                // ```
                //
                // The current implementation will begin optimization of plan #2 with the
                // a `maintain_order_above` state from after finishing plan 1.

                // If we don't know maintain order
                // Known: slice
                maintain_order_above = true;
            },
        }
    }
}
