//! This module creates predicates that can skip record batches of rows based on statistics about
//! that record batch.

use polars_core::prelude::{AnyValue, DataType, Scalar};
use polars_core::schema::Schema;
use polars_utils::aliases::PlIndexMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use super::super::evaluate::{constant_evaluate, into_column};
use super::super::{AExpr, IRBooleanFunction, IRFunctionExpr, Operator};
use crate::plans::aexpr::builder::IntoAExprBuilder;
use crate::plans::predicates::get_binary_expr_col_and_lv;
use crate::plans::{AExprBuilder, aexpr_to_leaf_names_iter, is_scalar_ae, rename_columns};

/// Return a new boolean expression determines whether a batch can be skipped based on min, max and
/// null count statistics.
///
/// This is conversative and may return `None` or `false` when an expression is not yet supported.
///
/// To evaluate, the expression it is given all the original column appended with `_min` and
/// `_max`. The `min` or `max` cannot be null and when they are null it is assumed they are not
/// known.
pub fn aexpr_to_skip_batch_predicate(
    e: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> Option<Node> {
    aexpr_to_skip_batch_predicate_rec(e, expr_arena, schema, 0)
}

fn does_dtype_have_sufficient_order(dtype: &DataType) -> bool {
    // Rules surrounding floats are really complicated. I should get around to that.
    !dtype.is_nested() && !dtype.is_float() && !dtype.is_null() && !dtype.is_categorical()
}

fn is_stat_defined(
    expr: impl IntoAExprBuilder,
    dtype: &DataType,
    arena: &mut Arena<AExpr>,
) -> AExprBuilder {
    let mut expr = expr.into_aexpr_builder();
    expr = expr.is_not_null(arena);
    if dtype.is_float() {
        let is_not_nan = expr.is_not_nan(arena);
        expr = expr.and(is_not_nan, arena);
    }
    expr
}

#[recursive::recursive]
fn aexpr_to_skip_batch_predicate_rec(
    e: Node,
    arena: &mut Arena<AExpr>,
    schema: &Schema,
    depth: usize,
) -> Option<Node> {
    use Operator as O;

    macro_rules! rec {
        ($node:expr) => {{ aexpr_to_skip_batch_predicate_rec($node, arena, schema, depth + 1) }};
    }
    macro_rules! lv_cases {
        (
            $lv:expr, $lv_node:expr,
            null: $null_case:expr,
            not_null: $non_null_case:expr $(,)?
        ) => {{
            if let Some(lv) = $lv {
                if lv.is_null() {
                    $null_case
                } else {
                    $non_null_case
                }
            } else {
                let lv_node = AExprBuilder::new_from_node($lv_node);

                let lv_is_null = lv_node.has_nulls(arena);
                let lv_not_null = lv_node.has_no_nulls(arena);

                let null_case = lv_is_null.and($null_case, arena);
                let non_null_case = lv_not_null.and($non_null_case, arena);

                null_case.or(non_null_case, arena).node()
            }
        }};
    }
    macro_rules! col {
        (len) => {{ col!(PlSmallStr::from_static("len")) }};
        ($name:expr) => {{ AExprBuilder::new_from_node(arena.add(AExpr::Column($name))) }};
        (min: $name:expr) => {{ col!(format_pl_smallstr!("{}_min", $name)) }};
        (max: $name:expr) => {{ col!(format_pl_smallstr!("{}_max", $name)) }};
        (null_count: $name:expr) => {{ col!(format_pl_smallstr!("{}_nc", $name)) }};
    }
    macro_rules! lv {
        ($lv:expr) => {{ AExprBuilder::lit_scalar(Scalar::from($lv), arena) }};
        (idx: $lv:expr) => {{ AExprBuilder::lit_scalar(Scalar::new_idxsize($lv), arena) }};
    }

    let specialized = (|| {
        if let Some(Some(lv)) = constant_evaluate(e, arena, schema, 0) {
            if let Some(av) = lv.to_any_value() {
                return match av {
                    AnyValue::Null => Some(lv!(true).node()),
                    AnyValue::Boolean(b) => Some(lv!(!b).node()),
                    _ => None,
                };
            }
        }

        match arena.get(e) {
            AExpr::Explode { .. } => None,
            AExpr::Column(_) => None,
            AExpr::Literal(_) => None,
            AExpr::BinaryExpr { left, op, right } => {
                let left = *left;
                let right = *right;

                match op {
                    O::Eq | O::EqValidity => {
                        let ((col, _), (lv, lv_node)) =
                            get_binary_expr_col_and_lv(left, right, arena, schema)?;
                        let dtype = schema.get(col)?;

                        if !does_dtype_have_sufficient_order(dtype) {
                            return None;
                        }

                        let op = *op;
                        let col = col.clone();

                        // col(A) == B -> {
                        //     null_count(A) == 0                              , if B.is_null(),
                        //     null_count(A) == LEN || min(A) > B || max(A) < B, if B.is_not_null(),
                        // }

                        Some(lv_cases!(
                            lv, lv_node,
                            null: {
                                if matches!(op, O::Eq) {
                                    lv!(false).node()
                                } else {
                                    let col_nc = col!(null_count: col);
                                    let idx_zero = lv!(idx: 0);
                                    col_nc.eq(idx_zero, arena).node()
                                }
                            },
                            not_null: {
                                let col_min = col!(min: col);
                                let col_max = col!(max: col);

                                let min_is_defined = is_stat_defined(col_min.node(), dtype, arena);
                                let max_is_defined = is_stat_defined(col_max.node(), dtype, arena);

                                let min_gt = col_min.gt(lv_node, arena);
                                let min_gt = min_gt.and(min_is_defined, arena);

                                let max_lt = col_max.lt(lv_node, arena);
                                let max_lt = max_lt.and(max_is_defined, arena);

                                let col_nc = col!(null_count: col);
                                let len = col!(len);
                                let all_nulls = col_nc.eq(len, arena);

                                all_nulls.or(min_gt, arena).or(max_lt, arena).node()
                            }
                        ))
                    },
                    O::NotEq | O::NotEqValidity => {
                        let ((col, _), (lv, lv_node)) =
                            get_binary_expr_col_and_lv(left, right, arena, schema)?;
                        let dtype = schema.get(col)?;

                        if !does_dtype_have_sufficient_order(dtype) {
                            return None;
                        }

                        let op = *op;
                        let col = col.clone();

                        // col(A) != B -> {
                        //     null_count(A) == LEN                            , if B.is_null(),
                        //     null_count(A) == 0 && min(A) == B && max(A) == B, if B.is_not_null(),
                        // }

                        Some(lv_cases!(
                            lv, lv_node,
                            null: {
                                if matches!(op, O::NotEq) {
                                    lv!(false).node()
                                } else {
                                    let col_nc = col!(null_count: col);
                                    let len = col!(len);
                                    col_nc.eq(len, arena).node()
                                }
                            },
                            not_null: {
                                let col_min = col!(min: col);
                                let col_max = col!(max: col);
                                let min_eq = col_min.eq(lv_node, arena);
                                let max_eq = col_max.eq(lv_node, arena);

                                let col_nc = col!(null_count: col);
                                let idx_zero = lv!(idx: 0);
                                let no_nulls = col_nc.eq(idx_zero, arena);

                                no_nulls.and(min_eq, arena).and(max_eq, arena).node()
                            }
                        ))
                    },
                    O::Lt | O::Gt | O::LtEq | O::GtEq => {
                        let ((col, col_node), (lv, lv_node)) =
                            get_binary_expr_col_and_lv(left, right, arena, schema)?;
                        let dtype = schema.get(col)?;

                        if !does_dtype_have_sufficient_order(dtype) {
                            return None;
                        }

                        let col_is_left = col_node == left;

                        let op = *op;
                        let col = col.clone();
                        let lv_may_be_null = lv.is_none_or(|lv| lv.is_null());

                        // If B is null, this is always true.
                        //
                        // col(A) < B       ~       B > col(A)  ->
                        //     null_count(A) == LEN || min(A) >= B
                        //
                        // col(A) > B       ~       B < col(A)  ->
                        //     null_count(A) == LEN || max(A) <= B
                        //
                        // col(A) <= B      ~       B >= col(A) ->
                        //     null_count(A) == LEN || min(A) > B
                        //
                        // col(A) >= B      ~       B <= col(A) ->
                        //     null_count(A) == LEN || max(A) < B

                        let stat = match (op, col_is_left) {
                            (O::Lt | O::LtEq, true) | (O::Gt | O::GtEq, false) => col!(min: col),
                            (O::Lt | O::LtEq, false) | (O::Gt | O::GtEq, true) => col!(max: col),
                            _ => unreachable!(),
                        };
                        let cmp_op = match (op, col_is_left) {
                            (O::Lt, true) | (O::Gt, false) => O::GtEq,
                            (O::Lt, false) | (O::Gt, true) => O::LtEq,

                            (O::LtEq, true) | (O::GtEq, false) => O::Gt,
                            (O::LtEq, false) | (O::GtEq, true) => O::Lt,

                            _ => unreachable!(),
                        };

                        let stat_is_defined = is_stat_defined(stat, dtype, arena);
                        let cmp_op = stat.binary_op(lv_node, cmp_op, arena);
                        let mut expr = stat_is_defined.and(cmp_op, arena);

                        if lv_may_be_null {
                            let has_nulls = lv_node.into_aexpr_builder().has_nulls(arena);
                            expr = has_nulls.or(expr, arena);
                        }
                        Some(expr.node())
                    },

                    O::And | O::LogicalAnd => match (rec!(left), rec!(right)) {
                        (Some(left), Some(right)) => {
                            Some(AExprBuilder::new_from_node(left).or(right, arena).node())
                        },
                        (Some(n), None) | (None, Some(n)) => Some(n),
                        (None, None) => None,
                    },
                    O::Or | O::LogicalOr => {
                        let left = rec!(left)?;
                        let right = rec!(right)?;
                        Some(AExprBuilder::new_from_node(left).and(right, arena).node())
                    },

                    O::Plus
                    | O::Minus
                    | O::Multiply
                    | O::Divide
                    | O::TrueDivide
                    | O::FloorDivide
                    | O::Modulus
                    | O::Xor => None,
                }
            },
            AExpr::Cast { .. } => None,
            AExpr::Sort { .. } => None,
            AExpr::Gather { .. } => None,
            AExpr::SortBy { .. } => None,
            AExpr::Filter { .. } => None,
            AExpr::Agg(..) => None,
            AExpr::Ternary { .. } => None,
            AExpr::AnonymousFunction { .. } => None,
            AExpr::Eval { .. } => None,
            AExpr::Function {
                input, function, ..
            } => match function {
                IRFunctionExpr::Boolean(f) => match f {
                    #[cfg(feature = "is_in")]
                    IRBooleanFunction::IsIn { nulls_equal } => {
                        if !is_scalar_ae(input[1].node(), arena) {
                            return None;
                        }

                        let nulls_equal = *nulls_equal;
                        let lv_node = input[1].node();
                        match (
                            into_column(input[0].node(), arena, schema, 0),
                            constant_evaluate(lv_node, arena, schema, 0),
                        ) {
                            (Some(col), Some(_)) => {
                                let dtype = schema.get(col)?;
                                if !does_dtype_have_sufficient_order(dtype) {
                                    return None;
                                }

                                // col(A).is_in([B1, ..., Bn]) ->
                                //      ([B1, ..., Bn].has_no_nulls() || null_count(A) == 0) &&
                                //      (
                                //          min(A) > max[B1, ..., Bn] ||
                                //          max(A) < min[B1, ..., Bn]
                                //      )
                                let col = col.clone();
                                let lv_node = lv_node.into_aexpr_builder();

                                let lv_node_exploded = lv_node.explode_skip_empty(arena);
                                let lv_min = lv_node_exploded.min(arena);
                                let lv_max = lv_node_exploded.max(arena);

                                let col_min = col!(min: col);
                                let col_max = col!(max: col);

                                let min_is_defined = is_stat_defined(col_min, dtype, arena);
                                let max_is_defined = is_stat_defined(col_max, dtype, arena);

                                let min_gt = col_min.gt(lv_max, arena);
                                let min_gt = min_is_defined.and(min_gt, arena);

                                let max_lt = col_max.lt(lv_min, arena);
                                let max_lt = max_is_defined.and(max_lt, arena);

                                let expr = min_gt.or(max_lt, arena);

                                let col_nc = col!(null_count: col);
                                let col_has_no_nulls = col_nc.has_no_nulls(arena);

                                let lv_has_not_nulls = lv_node_exploded.has_no_nulls(arena);
                                let null_case = lv_has_not_nulls.or(col_has_no_nulls, arena);

                                let min_max_is_in = null_case.and(expr, arena);

                                let col_nc = col!(null_count: col);

                                let min_is_max = col_min.eq(col_max, arena); // Eq so that (None == None) == None
                                let idx_zero = lv!(idx: 0);
                                let has_no_nulls = col_nc.eq(idx_zero, arena);

                                // The above case does always cover the fallback path. Since there
                                // is code that relies on the `min==max` always filtering normally,
                                // we add it here.
                                let exact_not_in =
                                    col_min.is_in(lv_node, nulls_equal, arena).not(arena);
                                let exact_not_in =
                                    min_is_max.and(has_no_nulls, arena).and(exact_not_in, arena);

                                Some(exact_not_in.or(min_max_is_in, arena).node())
                            },
                            _ => None,
                        }
                    },
                    IRBooleanFunction::IsNull => {
                        let col = into_column(input[0].node(), arena, schema, 0)?;

                        // col(A).is_null() -> null_count(A) == 0
                        let col_nc = col!(null_count: col);
                        let idx_zero = lv!(idx: 0);
                        Some(col_nc.eq(idx_zero, arena).node())
                    },
                    IRBooleanFunction::IsNotNull => {
                        let col = into_column(input[0].node(), arena, schema, 0)?;

                        // col(A).is_not_null() -> null_count(A) == LEN
                        let col_nc = col!(null_count: col);
                        let len = col!(len);
                        Some(col_nc.eq(len, arena).node())
                    },
                    #[cfg(feature = "is_between")]
                    IRBooleanFunction::IsBetween { closed } => {
                        let col = into_column(input[0].node(), arena, schema, 0)?;
                        let dtype = schema.get(col)?;

                        if !does_dtype_have_sufficient_order(dtype) {
                            return None;
                        }

                        // col(A).is_between(X, Y) ->
                        //     null_count(A) == LEN ||
                        //         min(A) >(=) Y ||
                        //         max(A) <(=) X

                        let left_node = input[1].node();
                        let right_node = input[2].node();

                        _ = constant_evaluate(left_node, arena, schema, 0)?;
                        _ = constant_evaluate(right_node, arena, schema, 0)?;

                        let col = col.clone();
                        let closed = *closed;

                        let lhs_no_nulls = left_node.into_aexpr_builder().has_no_nulls(arena);
                        let rhs_no_nulls = right_node.into_aexpr_builder().has_no_nulls(arena);

                        let col_min = col!(min: col);
                        let col_max = col!(max: col);

                        use polars_ops::series::ClosedInterval;
                        let (left, right) = match closed {
                            ClosedInterval::Both => (O::Lt, O::Gt),
                            ClosedInterval::Left => (O::Lt, O::GtEq),
                            ClosedInterval::Right => (O::LtEq, O::Gt),
                            ClosedInterval::None => (O::LtEq, O::GtEq),
                        };

                        let left = col_max.binary_op(left_node, left, arena);
                        let right = col_min.binary_op(right_node, right, arena);

                        let min_is_defined = is_stat_defined(col_min, dtype, arena);
                        let max_is_defined = is_stat_defined(col_max, dtype, arena);

                        let left = max_is_defined.and(left, arena);
                        let right = min_is_defined.and(right, arena);

                        let interval = left.or(right, arena);
                        Some(
                            lhs_no_nulls
                                .and(rhs_no_nulls, arena)
                                .and(interval, arena)
                                .node(),
                        )
                    },
                    _ => None,
                },
                _ => None,
            },
            AExpr::Window { .. } => None,
            AExpr::Slice { .. } => None,
            AExpr::Len => None,
        }
    })();

    if let Some(specialized) = specialized {
        return Some(specialized);
    }

    // If we don't have a specialized implementation we can check if the whole block is constant
    // and fill that value in. This is especially useful when filtering hive partitions which are
    // filtered using this expression and which set their min == max.
    //
    // Essentially, what this does is
    //     E -> all(col(A_min) == col(A_max) & col(A_nc) == 0 for A in LIVE(E)) & ~(E)

    let live_columns = PlIndexMap::from_iter(aexpr_to_leaf_names_iter(e, arena).map(|col| {
        let min_name = format_pl_smallstr!("{col}_min");
        (col, min_name)
    }));

    // We cannot do proper equalities for these.
    if live_columns
        .iter()
        .any(|(c, _)| schema.get(c).is_none_or(|dt| dt.is_categorical()))
    {
        return None;
    }

    // Rename all uses of column names with the min value.
    let expr = rename_columns(e, arena, &live_columns);
    let mut expr = expr.into_aexpr_builder().not(arena);
    for col in live_columns.keys() {
        let col_min = col!(min: col);
        let col_max = col!(max: col);
        let col_nc = col!(null_count: col);

        let min_is_max = col_min.eq(col_max, arena); // Eq so that (None == None) == None
        let idx_zero = lv!(idx: 0);
        let has_no_nulls = col_nc.eq(idx_zero, arena);

        expr = min_is_max.and(has_no_nulls, arena).and(expr, arena);
    }
    Some(expr.node())
}
