use polars_utils::unitvec;

use super::*;

// Check if the aggregate expression depends on the order of the data.
fn is_order_independent_agg(agg: &IRAggExpr) -> bool {
    match agg {
        IRAggExpr::Min { .. } => true,
        IRAggExpr::Max { .. } => true,
        IRAggExpr::Median(_) => true,
        IRAggExpr::NUnique(_) => true,
        IRAggExpr::First(_) => false,
        IRAggExpr::Last(_) => false,
        IRAggExpr::Mean(_) => true,
        IRAggExpr::Implode(_) => false,
        IRAggExpr::Quantile { .. } => true,
        IRAggExpr::Sum(_) => true,
        IRAggExpr::Count(_, _) => true,
        IRAggExpr::Std(_, _) => true,
        IRAggExpr::Var(_, _) => true,
        IRAggExpr::AggGroups(_) => false,
    }
}

// Check if the expression is a data source (e.g., Column), or keeps the underlying
// data set unmodified, i.e., set(f(data)) == set(data).
// Not exhaustive, e.g. changing the order would fall in here
fn is_source_or_set_invariant(ae: &AExpr) -> bool {
    matches!(ae, AExpr::Column(_))
}

// Check if the given node, or, recursively, any of its input nodes contains an
// order_dependent operation. Return true if the output does not depend on ordering.
// Can give false negatives.
fn is_order_independent_rec(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut contains_safe_aggregate = false;

    let mut stack = unitvec![];
    stack.push(node);

    while let Some(node) = stack.pop() {
        let ae = expr_arena.get(node);

        // we need at least one 'safe' (order_independent) aggregation
        if let AExpr::Agg(agg) = ae {
            if is_order_independent_agg(agg) {
                contains_safe_aggregate = true;
            } else {
                return false;
            }
        } else {
            // intermediate data modification is not allowed, hence
            // only data sources or set-invariant expressions are allowed
            if !is_source_or_set_invariant(ae) {
                return false;
            }
        }

        ae.inputs_rev(&mut stack);
    }

    contains_safe_aggregate
}

// Not exhaustive - can give false negatives.
pub(crate) fn all_order_independent<'a, N>(nodes: &'a [N], expr_arena: &Arena<AExpr>) -> bool
where
    Node: From<&'a N>,
{
    nodes
        .iter()
        .all(|n| is_order_independent_rec(n.into(), expr_arena))
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

                maintain_order_above =
                    !(all_elementwise(keys, expr_arena) && all_order_independent(aggs, expr_arena));
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
