//! This module creates predicates that can skip record batches of rows based on statistics about
//! that record batch.

use polars_core::prelude::{AnyValue, DataType, Scalar};
use polars_core::schema::Schema;
use polars_utils::aliases::PlIndexMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use super::super::evaluate::{constant_evaluate, into_column};
use super::super::{AExpr, BooleanFunction, Operator, OutputName};
use crate::dsl::FunctionExpr;
use crate::plans::predicates::get_binary_expr_col_and_lv;
use crate::plans::{ExprIR, LiteralValue, aexpr_to_leaf_names_iter, is_scalar_ae, rename_columns};
use crate::prelude::FunctionOptions;

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

#[recursive::recursive]
fn aexpr_to_skip_batch_predicate_rec(
    e: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
    depth: usize,
) -> Option<Node> {
    use Operator as O;

    macro_rules! rec {
        ($node:expr) => {{ aexpr_to_skip_batch_predicate_rec($node, expr_arena, schema, depth + 1) }};
    }
    macro_rules! and {
        ($l:expr, $($r:expr),+ $(,)?) => {{
            let node = $l;
            $(
            let node = expr_arena.add(AExpr::BinaryExpr {
                left: node,
                op: O::LogicalAnd,
                right: $r,
            });
            )+
            node
        }}
    }
    macro_rules! or {
        ($l:expr, $($r:expr),+ $(,)?) => {{
            let node = $l;
            $(
            let node = expr_arena.add(AExpr::BinaryExpr {
                left: node,
                op: O::LogicalOr,
                right: $r,
            });
            )+
            node
        }}
    }
    macro_rules! binexpr {
        (op: $op:expr, $l:expr, $r:expr) => {{
            expr_arena.add(AExpr::BinaryExpr {
                left: $l,
                op: $op,
                right: $r,
            })
        }};
        ($op:ident, $l:expr, $r:expr) => {{ binexpr!(op: O::$op, $l, $r) }};
    }
    macro_rules! lt {
        ($l:expr, $r:expr) => {{ binexpr!(Lt, $l, $r) }};
    }
    macro_rules! gt {
        ($l:expr, $r:expr) => {{ binexpr!(Gt, $l, $r) }};
    }
    macro_rules! eq {
        ($l:expr, $r:expr) => {{ binexpr!(Eq, $l, $r) }};
    }
    macro_rules! null_count {
        ($i:expr) => {{
            expr_arena.add(AExpr::Function {
                input: vec![ExprIR::new($i, OutputName::Alias(PlSmallStr::EMPTY))],
                function: FunctionExpr::NullCount,
                options: FunctionOptions::default(),
            })
        }};
    }
    macro_rules! has_no_nulls {
        ($i:expr) => {{
            let expr = null_count!($i);
            let idx_zero = lv!(0);
            eq!(expr, idx_zero)
        }};
    }
    macro_rules! has_nulls {
        ($i:expr) => {{
            let expr = null_count!($i);
            let idx_zero = lv!(0);
            gt!(expr, idx_zero)
        }};
    }
    macro_rules! is_stat_defined {
        ($i:expr, $dtype:expr) => {{
            let mut expr = expr_arena.add(AExpr::Function {
                input: vec![ExprIR::new($i, OutputName::Alias(PlSmallStr::EMPTY))],
                function: FunctionExpr::Boolean(BooleanFunction::IsNotNull),
                options: FunctionOptions::default(),
            });

            if $dtype.is_float() {
                let is_not_nan = expr_arena.add(AExpr::Function {
                    input: vec![ExprIR::new($i, OutputName::Alias(PlSmallStr::EMPTY))],
                    function: FunctionExpr::Boolean(BooleanFunction::IsNotNan),
                    options: FunctionOptions::default(),
                });
                expr = and!(is_not_nan, expr);
            }

            expr
        }};
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
                let lv_is_null = has_nulls!($lv_node);
                let lv_not_null = has_no_nulls!($lv_node);

                let null_case = $null_case;
                let null_case = and!(lv_is_null, null_case);
                let non_null_case = $non_null_case;
                let non_null_case = and!(lv_not_null, non_null_case);

                or!(null_case, non_null_case)
            }
        }};
    }
    macro_rules! col {
        (len) => {{ col!(PlSmallStr::from_static("len")) }};
        ($name:expr) => {{ expr_arena.add(AExpr::Column($name)) }};
        (min: $name:expr) => {{ col!(format_pl_smallstr!("{}_min", $name)) }};
        (max: $name:expr) => {{ col!(format_pl_smallstr!("{}_max", $name)) }};
        (null_count: $name:expr) => {{ col!(format_pl_smallstr!("{}_nc", $name)) }};
    }
    macro_rules! lv {
        ($lv:expr) => {{ expr_arena.add(AExpr::Literal(Scalar::from($lv).into())) }};
        (idx: $lv:expr) => {{ expr_arena.add(AExpr::Literal(LiteralValue::new_idxsize($lv))) }};
    }

    let specialized = (|| {
        if let Some(Some(lv)) = constant_evaluate(e, expr_arena, schema, 0) {
            if let Some(av) = lv.to_any_value() {
                return match av {
                    AnyValue::Null => Some(lv!(true)),
                    AnyValue::Boolean(b) => Some(lv!(!b)),
                    _ => None,
                };
            }
        }

        match expr_arena.get(e) {
            AExpr::Explode { .. } => None,
            AExpr::Alias(_, _) => None,
            AExpr::Column(_) => None,
            AExpr::Literal(_) => None,
            AExpr::BinaryExpr { left, op, right } => {
                let left = *left;
                let right = *right;

                match op {
                    O::Eq | O::EqValidity => {
                        let ((col, _), (lv, lv_node)) =
                            get_binary_expr_col_and_lv(left, right, expr_arena, schema)?;
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
                                    lv!(false)
                                } else {
                                    let col_nc = col!(null_count: col);
                                    let idx_zero = lv!(idx: 0);
                                    eq!(col_nc, idx_zero)
                                }
                            },
                            not_null: {
                                let col_min = col!(min: col);
                                let col_max = col!(max: col);

                                let min_is_defined = is_stat_defined!(col_min, dtype);
                                let max_is_defined = is_stat_defined!(col_max, dtype);

                                let min_gt = gt!(col_min, lv_node);
                                let min_gt = and!(min_is_defined, min_gt);

                                let max_lt = lt!(col_max, lv_node);
                                let max_lt = and!(max_is_defined, max_lt);

                                let col_nc = col!(null_count: col);
                                let len = col!(len);
                                let all_nulls = eq!(col_nc, len);

                                or!(all_nulls, min_gt, max_lt)
                            }
                        ))
                    },
                    O::NotEq | O::NotEqValidity => {
                        let ((col, _), (lv, lv_node)) =
                            get_binary_expr_col_and_lv(left, right, expr_arena, schema)?;
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
                                    lv!(false)
                                } else {
                                    let col_nc = col!(null_count: col);
                                    let len = col!(len);
                                    eq!(col_nc, len)
                                }
                            },
                            not_null: {
                                let col_min = col!(min: col);
                                let col_max = col!(max: col);
                                let min_eq = eq!(col_min, lv_node);
                                let max_eq = eq!(col_max, lv_node);

                                let col_nc = col!(null_count: col);
                                let idx_zero = lv!(idx: 0);
                                let no_nulls = eq!(col_nc, idx_zero);

                                and!(no_nulls, min_eq, max_eq)
                            }
                        ))
                    },
                    O::Lt | O::Gt | O::LtEq | O::GtEq => {
                        let ((col, col_node), (lv, lv_node)) =
                            get_binary_expr_col_and_lv(left, right, expr_arena, schema)?;
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

                        let stat_is_defined = is_stat_defined!(stat, dtype);
                        let cmp_op = binexpr!(op: cmp_op, stat, lv_node);
                        let mut expr = and!(stat_is_defined, cmp_op);

                        if lv_may_be_null {
                            let has_nulls = has_nulls!(lv_node);
                            expr = or!(has_nulls, expr);
                        }
                        Some(expr)
                    },

                    O::And | O::LogicalAnd => match (rec!(left), rec!(right)) {
                        (Some(left), Some(right)) => Some(or!(left, right)),
                        (Some(n), None) | (None, Some(n)) => Some(n),
                        (None, None) => None,
                    },
                    O::Or | O::LogicalOr => {
                        let left = rec!(left)?;
                        let right = rec!(right)?;
                        Some(and!(left, right))
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
            AExpr::Function {
                input, function, ..
            } => match function {
                FunctionExpr::Boolean(f) => match f {
                    #[cfg(feature = "is_in")]
                    BooleanFunction::IsIn { nulls_equal } => {
                        if !is_scalar_ae(input[1].node(), expr_arena) {
                            return None;
                        }

                        let nulls_equal = *nulls_equal;
                        let lv_node = input[1].node();
                        match (
                            into_column(input[0].node(), expr_arena, schema, 0),
                            constant_evaluate(lv_node, expr_arena, schema, 0),
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

                                let lv_node_exploded = expr_arena.add(AExpr::Explode {
                                    expr: lv_node,
                                    skip_empty: true,
                                });
                                let lv_min =
                                    expr_arena.add(AExpr::Agg(crate::plans::IRAggExpr::Min {
                                        input: lv_node_exploded,
                                        propagate_nans: true,
                                    }));
                                let lv_max =
                                    expr_arena.add(AExpr::Agg(crate::plans::IRAggExpr::Max {
                                        input: lv_node_exploded,
                                        propagate_nans: true,
                                    }));

                                let col_min = col!(min: col);
                                let col_max = col!(max: col);

                                let min_is_defined = is_stat_defined!(col_min, dtype);
                                let max_is_defined = is_stat_defined!(col_max, dtype);

                                let min_gt = gt!(col_min, lv_max);
                                let min_gt = and!(min_is_defined, min_gt);

                                let max_lt = lt!(col_max, lv_min);
                                let max_lt = and!(max_is_defined, max_lt);

                                let expr = or!(min_gt, max_lt);

                                let col_nc = col!(null_count: col);
                                let idx_zero = lv!(idx: 0);
                                let col_has_no_nulls = eq!(col_nc, idx_zero);

                                let lv_has_not_nulls = has_no_nulls!(lv_node_exploded);
                                let null_case = or!(lv_has_not_nulls, col_has_no_nulls);

                                let min_max_is_in = and!(null_case, expr);

                                let col_nc = col!(null_count: col);

                                let min_is_max = binexpr!(Eq, col_min, col_max); // Eq so that (None == None) == None
                                let idx_zero = lv!(idx: 0);
                                let has_no_nulls = eq!(col_nc, idx_zero);

                                // The above case does always cover the fallback path. Since there
                                // is code that relies on the `min==max` always filtering normally,
                                // we add it here.
                                let exact_not_in = expr_arena.add(AExpr::Function {
                                    input: vec![
                                        ExprIR::new(col_min, OutputName::Alias(PlSmallStr::EMPTY)),
                                        ExprIR::new(lv_node, OutputName::Alias(PlSmallStr::EMPTY)),
                                    ],
                                    function: FunctionExpr::Boolean(BooleanFunction::IsIn {
                                        nulls_equal,
                                    }),
                                    options: BooleanFunction::IsIn { nulls_equal }
                                        .function_options(),
                                });
                                let exact_not_in = expr_arena.add(AExpr::Function {
                                    input: vec![ExprIR::new(
                                        exact_not_in,
                                        OutputName::Alias(PlSmallStr::EMPTY),
                                    )],
                                    function: FunctionExpr::Boolean(BooleanFunction::Not),
                                    options: BooleanFunction::Not.function_options(),
                                });
                                let exact_not_in = and!(min_is_max, has_no_nulls, exact_not_in);

                                Some(or!(exact_not_in, min_max_is_in))
                            },
                            _ => None,
                        }
                    },
                    BooleanFunction::IsNull => {
                        let col = into_column(input[0].node(), expr_arena, schema, 0)?;

                        // col(A).is_null() -> null_count(A) == 0
                        let col_nc = col!(null_count: col);
                        let idx_zero = lv!(idx: 0);
                        Some(eq!(col_nc, idx_zero))
                    },
                    BooleanFunction::IsNotNull => {
                        let col = into_column(input[0].node(), expr_arena, schema, 0)?;

                        // col(A).is_not_null() -> null_count(A) == LEN
                        let col_nc = col!(null_count: col);
                        let len = col!(len);
                        Some(eq!(col_nc, len))
                    },
                    #[cfg(feature = "is_between")]
                    BooleanFunction::IsBetween { closed } => {
                        let col = into_column(input[0].node(), expr_arena, schema, 0)?;
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

                        _ = constant_evaluate(left_node, expr_arena, schema, 0)?;
                        _ = constant_evaluate(right_node, expr_arena, schema, 0)?;

                        let col = col.clone();
                        let closed = *closed;

                        let lhs_no_nulls = has_no_nulls!(left_node);
                        let rhs_no_nulls = has_no_nulls!(right_node);

                        let col_min = col!(min: col);
                        let col_max = col!(max: col);

                        use polars_ops::series::ClosedInterval;
                        let (left, right) = match closed {
                            ClosedInterval::Both => (O::Lt, O::Gt),
                            ClosedInterval::Left => (O::Lt, O::GtEq),
                            ClosedInterval::Right => (O::LtEq, O::Gt),
                            ClosedInterval::None => (O::LtEq, O::GtEq),
                        };

                        let left = binexpr!(op: left, col_max, left_node);
                        let right = binexpr!(op: right, col_min, right_node);

                        let min_is_defined = is_stat_defined!(col_min, dtype);
                        let max_is_defined = is_stat_defined!(col_max, dtype);

                        let left = and!(max_is_defined, left);
                        let right = and!(min_is_defined, right);

                        let interval = or!(left, right);
                        Some(and!(lhs_no_nulls, rhs_no_nulls, interval))
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

    let live_columns = PlIndexMap::from_iter(aexpr_to_leaf_names_iter(e, expr_arena).map(|col| {
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
    let expr = rename_columns(e, expr_arena, &live_columns);
    let mut expr = expr_arena.add(AExpr::Function {
        input: vec![ExprIR::new(expr, OutputName::Alias(PlSmallStr::EMPTY))],
        function: FunctionExpr::Boolean(BooleanFunction::Not),
        options: FunctionOptions::elementwise(),
    });
    for col in live_columns.keys() {
        let col_min = col!(min: col);
        let col_max = col!(max: col);
        let col_nc = col!(null_count: col);

        let min_is_max = binexpr!(Eq, col_min, col_max); // Eq so that (None == None) == None
        let idx_zero = lv!(idx: 0);
        let has_no_nulls = eq!(col_nc, idx_zero);

        expr = and!(min_is_max, has_no_nulls, expr);
    }
    Some(expr)
}
