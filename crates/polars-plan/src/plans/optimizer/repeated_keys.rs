use std::hash::Hash;

use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::{OptimizationRule, OptimizeExprContext};
use crate::plans::{AExpr, CanonicalExprMap};
use crate::prelude::*;

/// Removes keys that repeat an earlier key from operations whose result does not change when a
/// key is repeated, such as sorting, `unique` and window partitions.
pub struct RemoveRepeatedKeys {}

impl OptimizationRule for RemoveRepeatedKeys {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let mut ir = lp_arena.get(node).clone();
        let changed = match &mut ir {
            IR::Sort {
                by_column,
                sort_options,
                ..
            } => first_expr_occurrences(by_column.iter().map(|e| e.node()), expr_arena)
                .map(|keep| retain_sort_keys(by_column, sort_options, &keep))
                .is_some(),
            IR::Distinct { options, .. } => {
                if let Some(subset) = &mut options.subset
                    && let Some(keep) = first_occurrences(subset.iter().map(Some))
                {
                    *subset = gather(subset, &keep).into();
                    true
                } else {
                    false
                }
            },
            #[cfg(feature = "asof_join")]
            IR::Join { options, .. } => {
                if let JoinType::AsOf(asof) = &options.args.how
                    && let (Some(left_by), Some(right_by)) = (&asof.left_by, &asof.right_by)
                    && let Some(keep) = first_occurrences(left_by.iter().zip(right_by).map(Some))
                {
                    let mut asof = asof.clone();
                    asof.left_by = Some(gather(left_by, &keep));
                    asof.right_by = Some(gather(right_by, &keep));
                    Arc::make_mut(options).args.how = JoinType::AsOf(asof);
                    true
                } else {
                    false
                }
            },
            _ => false,
        };
        Ok(changed.then_some(ir))
    }

    fn optimize_expr(
        &mut self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        _schema: &Schema,
        _ctx: OptimizeExprContext,
    ) -> PolarsResult<Option<AExpr>> {
        let mut ae = expr_arena.get(expr_node).clone();
        let changed = match &mut ae {
            AExpr::SortBy {
                by, sort_options, ..
            } => first_expr_occurrences(by.iter().copied(), expr_arena)
                .map(|keep| retain_sort_keys(by, sort_options, &keep))
                .is_some(),
            #[cfg(feature = "top_k")]
            AExpr::Function {
                input,
                function: IRFunctionExpr::TopKBy { descending },
                ..
            } => {
                // The inputs are the values, `k` and then the keys.
                let keys = input.split_off(2.min(input.len()));
                if let Some(keep) =
                    first_expr_occurrences(keys.iter().map(|e| e.node()), expr_arena)
                {
                    *descending = gather(descending, &keep);
                    input.extend(gather(&keys, &keep));
                    true
                } else {
                    false
                }
            },
            AExpr::Over {
                partition_by,
                order_by,
                ..
            } => {
                let mut changed = false;
                if let Some(keep) = first_expr_occurrences(partition_by.iter().copied(), expr_arena)
                {
                    *partition_by = gather(partition_by, &keep);
                    changed = true;
                }
                if let Some((node, options)) = order_by
                    && let Some(new_order_by) =
                        retain_row_encode_keys(*node, *options, expr_arena, |keys, arena| {
                            first_expr_occurrences(keys.iter().map(|e| e.node()), arena)
                        })
                {
                    *order_by = Some(new_order_by);
                    changed = true;
                }
                changed
            },
            _ => false,
        };
        Ok(changed.then_some(ae))
    }
}

/// Keeps the keys of an ordered row encoding, which is how a window combines several `order_by`
/// keys, at the positions `select` returns. `None` if nothing changes.
pub(super) fn retain_row_encode_keys(
    node: Node,
    options: SortOptions,
    expr_arena: &mut Arena<AExpr>,
    select: impl FnOnce(&[ExprIR], &Arena<AExpr>) -> Option<Vec<usize>>,
) -> Option<(Node, SortOptions)> {
    let mut encode = expr_arena.get(node).clone();
    let AExpr::Function {
        input,
        function:
            IRFunctionExpr::RowEncode(
                dtypes,
                RowEncodingVariant::Ordered {
                    descending,
                    nulls_last,
                    broadcast_nulls,
                },
            ),
        ..
    } = &mut encode
    else {
        return None;
    };
    let keep = select(input, expr_arena).filter(|keep| !keep.is_empty())?;
    *input = gather(input, &keep);
    *dtypes = gather(dtypes, &keep);
    for flags in [&mut *descending, &mut *nulls_last].into_iter().flatten() {
        *flags = gather(flags, &keep);
    }

    // Without null broadcasting a single key sorts the same as its row encoding.
    if let [key] = &input[..]
        && !broadcast_nulls.unwrap_or(false)
    {
        let flag = |flags: &Option<Vec<bool>>| flags.as_ref().is_some_and(|f| f[0]);
        let options = SortOptions {
            descending: flag(descending) ^ options.descending,
            nulls_last: flag(nulls_last) ^ options.descending,
            ..options
        };
        return Some((key.node(), options));
    }
    Some((expr_arena.add(encode), options))
}

pub(super) fn retain_sort_keys<T: Clone>(
    by: &mut Vec<T>,
    options: &mut SortMultipleOptions,
    keep: &[usize],
) {
    *by = gather(by, keep);
    options.descending = gather(&options.descending, keep);
    options.nulls_last = gather(&options.nulls_last, keep);
}

/// Like [`first_occurrences`] for expressions. Non-deterministic expressions are never repeats.
pub(super) fn first_expr_occurrences(
    nodes: impl IntoIterator<Item = Node>,
    expr_arena: &Arena<AExpr>,
) -> Option<Vec<usize>> {
    let mut map = CanonicalExprMap::new();
    first_occurrences(nodes.into_iter().map(|node| {
        let id = map.resolve(node, expr_arena);
        (!map.is_nondeterministic(id)).then_some(id)
    }))
}

/// Returns the positions of the keys that do not repeat an earlier key, or `None` if no key
/// repeats. `None` keys are never repeats.
fn first_occurrences<K: Hash + Eq>(
    keys: impl IntoIterator<Item = Option<K>>,
) -> Option<Vec<usize>> {
    let mut seen = PlIndexSet::new();
    let mut num_keys = 0;
    let keep: Vec<usize> = keys
        .into_iter()
        .enumerate()
        .inspect(|_| num_keys += 1)
        .filter_map(|(i, key)| key.is_none_or(|key| seen.insert(key)).then_some(i))
        .collect();
    (keep.len() < num_keys).then_some(keep)
}

fn gather<T: Clone>(values: &[T], keep: &[usize]) -> Vec<T> {
    keep.iter().map(|&i| values[i].clone()).collect()
}
