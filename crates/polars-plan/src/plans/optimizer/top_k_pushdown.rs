//! Copying a top-k onto the preserved side of a left, right or cross join below it.
//!
//! `top_k(A ⟕ B)` equals `top_k(top_k(A) ⟕ B)` when every sort key comes from `A`. Every row
//! of `A` yields at least one output row with the same key values, so the `k` best rows of `A`
//! yield at least `k` output rows, each at least as good as any row derived from a dropped row
//! of `A`. A row of `B` that only matched dropped rows vanishes either way. A right join is the
//! mirror image, and a cross join preserves both sides.

use polars_core::prelude::{IdxSize, PlIndexSet};
use polars_defs::join::JoinValidation;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;

use crate::plans::schema::join_right_output_names;
use crate::plans::{
    AExpr, ArenaExprIter, ArenaLpIter, ExprIR, ExprPushdownGroup, IR, IRFunctionExpr,
    JoinTypeOptionsIR, aexpr_to_leaf_names_iter, is_elementwise_rec,
    is_inherently_nondeterministic,
};
use crate::prelude::{JoinType, MaintainOrderJoin};

/// Copy every top-k onto the input of the lowest left, right or cross join that preserves its
/// keys.
pub(super) fn push_down_top_k(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    maintain_errors: bool,
) {
    // A cached subtree is visited once per parent.
    let sorts: PlIndexSet<Node> = ir_arena
        .iter(root)
        .filter(|(_, ir)| matches!(ir, IR::Sort { slice: Some(_), .. }))
        .map(|(node, _)| node)
        .collect();
    for sort in sorts {
        copy_top_k(sort, root, ir_arena, expr_arena, maintain_errors);
    }
}

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

impl Side {
    fn take<T>(self, left: T, right: T) -> T {
        match self {
            Side::Left => left,
            Side::Right => right,
        }
    }
}

fn copy_top_k(
    sort: Node,
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    maintain_errors: bool,
) {
    let IR::Sort {
        input,
        by_column,
        slice: Some((offset, len, dyn_pred)),
        sort_options,
    } = ir_arena.get(sort)
    else {
        unreachable!()
    };
    // A slice without a length has length `IdxSize::MAX`.
    let Some(k) = usize::try_from(*offset)
        .ok()
        .and_then(|offset| offset.checked_add(*len))
        .filter(|&k| k < IdxSize::MAX as usize)
    else {
        return;
    };
    // The copy evaluates the keys again, on the preserved side, whose rows are the output's
    // without duplicates.
    if !by_column.iter().all(|key| {
        is_elementwise_rec(key.node(), expr_arena)
            && !is_inherently_nondeterministic(key.node(), expr_arena)
    }) {
        return;
    }
    let are_key_exprs_fallible = by_column.iter().any(|key| {
        expr_arena
            .iter(key.node())
            .any(|(_, ae)| ae.is_fallible_top_level(expr_arena))
    });
    let key_columns: PlIndexSet<PlSmallStr> = by_column
        .iter()
        .flat_map(|key| aexpr_to_leaf_names_iter(key.node(), expr_arena).cloned())
        .collect();
    if key_columns.is_empty() {
        return;
    }
    let Some((join, side, input)) = lowest_preserving_join(
        *input,
        &key_columns,
        sort_options.maintain_order,
        are_key_exprs_fallible,
        ir_arena,
        expr_arena,
        maintain_errors,
    ) else {
        return;
    };

    let move_dyn_pred = dyn_pred
        .as_ref()
        .is_some_and(|pred| is_read_only_below(pred.id(), input, root, ir_arena, expr_arena));

    let copy = ir_arena.add(IR::Sort {
        input,
        by_column: by_column.clone(),
        slice: Some((0, k, None)),
        sort_options: sort_options.clone(),
    });
    // The walk passes no `Cache`, so `join` has no other parent.
    let IR::Join {
        input_left,
        input_right,
        ..
    } = ir_arena.get_mut(join)
    else {
        unreachable!()
    };
    *side.take(input_left, input_right) = copy;

    if move_dyn_pred {
        swap_dyn_preds(sort, copy, ir_arena);
    }
}

/// Swap the dynamic predicates of n1 and n2.
fn swap_dyn_preds(n1: Node, n2: Node, ir_arena: &mut Arena<IR>) {
    let [
        IR::Sort {
            slice: Some((_, _, dp1)),
            ..
        },
        IR::Sort {
            slice: Some((_, _, dp2)),
            ..
        },
    ] = ir_arena.get_disjoint_mut([n1, n2])
    else {
        unreachable!()
    };
    std::mem::swap(dp1, dp2);
}

/// Whether every node of the plan at `root` that reads the dynamic predicate `id` lies below
/// `node`, without a `Cache` in between.
fn is_read_only_below(
    id: &UniqueId,
    node: Node,
    root: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> bool {
    let mut below = PlIndexSet::default();
    let mut stack = vec![node];
    while let Some(node) = stack.pop() {
        let ir = ir_arena.get(node);
        if below.insert(node) && !matches!(ir, IR::Cache { .. }) {
            stack.extend(ir.inputs());
        }
    }
    let reads_dyn_pred = |ir: &IR| {
        ir.exprs().any(|e| {
            expr_arena.iter(e.node()).any(|(_, ae)| {
                matches!(
                    ae,
                    AExpr::Function {
                        function: IRFunctionExpr::DynamicPred { pred }
                            | IRFunctionExpr::DynamicSkipBatch { pred },
                        ..
                    } if pred.id().as_ref() == Some(id)
                )
            })
        })
    };
    ir_arena
        .iter(root)
        .all(|(node, ir)| below.contains(&node) || !reads_dyn_pred(ir))
}

/// Walk down from `node` through row-wise projections and joins that preserve `key_columns`,
/// and return the lowest such join with its preserved side and that side's input.
fn lowest_preserving_join(
    mut node: Node,
    key_columns: &PlIndexSet<PlSmallStr>,
    is_stable: bool,
    are_key_exprs_fallible: bool,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    maintain_errors: bool,
) -> Option<(Node, Side, Node)> {
    let exprs_are_pushable = |exprs: &[ExprIR]| {
        exprs
            .iter()
            .all(|e| expr_is_pushable(e.node(), expr_arena, maintain_errors))
    };
    let mut lowest = None;
    loop {
        node = match ir_arena.get(node) {
            IR::SimpleProjection { input, .. } => *input,
            IR::Select { input, expr, .. }
                if passes_keys_through(expr, key_columns, false, expr_arena)
                    && exprs_are_pushable(expr) =>
            {
                *input
            },
            IR::HStack { input, exprs, .. }
                if passes_keys_through(exprs, key_columns, true, expr_arena)
                    && exprs_are_pushable(exprs) =>
            {
                *input
            },
            join @ IR::Join {
                input_left,
                input_right,
                ..
            } => {
                let Some(side) = preserved_side(
                    join,
                    key_columns,
                    is_stable,
                    are_key_exprs_fallible,
                    ir_arena,
                    expr_arena,
                    maintain_errors,
                ) else {
                    break;
                };
                let input = side.take(*input_left, *input_right);
                lowest = Some((node, side, input));
                input
            },
            _ => break,
        };
    }
    lowest
}

/// Whether the expression is elementwise, and infallible if `maintain_errors`.
fn expr_is_pushable(node: Node, expr_arena: &Arena<AExpr>, maintain_errors: bool) -> bool {
    let mut group = ExprPushdownGroup::Pushable;
    group.update_with_expr_rec(expr_arena.get(node), expr_arena, None);
    !group.blocks_pushdown(maintain_errors)
}

/// Whether `exprs` output every key column as that same column. With `is_hstack`, a key column
/// that no expression outputs passes through as well.
fn passes_keys_through(
    exprs: &[ExprIR],
    key_columns: &PlIndexSet<PlSmallStr>,
    is_hstack: bool,
    expr_arena: &Arena<AExpr>,
) -> bool {
    key_columns.iter().all(
        |name| match exprs.iter().find(|e| e.output_name() == name) {
            Some(e) => e.plain_column(expr_arena) == Some(name),
            None => is_hstack,
        },
    )
}

/// Get the side of `join` that keeps all its rows and supplies `key_columns` under their own
/// names, if a top-k may be copied onto it.
fn preserved_side(
    join: &IR,
    key_columns: &PlIndexSet<PlSmallStr>,
    is_stable: bool,
    are_key_exprs_fallible: bool,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    maintain_errors: bool,
) -> Option<Side> {
    use MaintainOrderJoin as MOJ;
    let IR::Join {
        input_left,
        input_right,
        options,
        ..
    } = join
    else {
        unreachable!()
    };
    let args = &options.args;
    let left_schema = ir_arena.get(*input_left).schema(ir_arena);
    let right_schema = ir_arena.get(*input_right).schema(ir_arena);

    // A slice drops rows, and fewer rows could hide a validation error.
    if !matches!(
        options.options,
        JoinTypeOptionsIR::Equi {
            fused_predicate: None,
            ..
        }
    ) || args.slice.is_some()
        || !matches!(args.validation, JoinValidation::ManyToMany)
    {
        return None;
    }

    // Left columns are only renamed by a coalescing right join, which drops its left keys.
    let key_is_from_left = || key_columns.iter().all(|name| left_schema.contains(name));
    let key_is_from_right = || {
        let Ok(output_names) = join_right_output_names(&left_schema, &right_schema, options) else {
            return false;
        };
        key_columns.iter().all(|name| {
            right_schema
                .index_of(name)
                .is_some_and(|i| output_names[i].as_ref() == Some(name))
        })
    };
    let side = match args.how {
        JoinType::Left if key_is_from_left() => Side::Left,
        JoinType::Right if key_is_from_right() => Side::Right,
        // An empty other side drops every row, so the copy could raise an error that the original
        // does not.
        JoinType::Cross if are_key_exprs_fallible => return None,
        JoinType::Cross if key_is_from_left() => Side::Left,
        JoinType::Cross if key_is_from_right() => Side::Right,
        _ => return None,
    };
    let is_order_compatible = match (side, args.maintain_order) {
        (_, MOJ::None) => true,
        // The other side decides the order of ties.
        (Side::Left, MOJ::Right) | (Side::Right, MOJ::Left) => !is_stable,
        (Side::Left, MOJ::Left | MOJ::LeftRight | MOJ::RightLeft)
        | (Side::Right, MOJ::Right | MOJ::LeftRight | MOJ::RightLeft) => false,
    };
    let mut side_keys = side.take(options.options.left_on(), options.options.right_on());
    (is_order_compatible
        && side_keys.all(|key| expr_is_pushable(key.node(), expr_arena, maintain_errors)))
    .then_some(side)
}
