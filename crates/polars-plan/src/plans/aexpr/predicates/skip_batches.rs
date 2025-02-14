//! This module creates predicates that can skip record batches of rows based on statistics about
//! that record batch.

use polars_core::prelude::{AnyValue, Scalar};
use polars_core::schema::Schema;
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use super::super::evaluate::{constant_evaluate, into_column};
use super::super::{AExpr, BooleanFunction, Operator, OutputName};
use crate::dsl::FunctionExpr;
use crate::plans::predicates::get_binary_expr_col_and_lv;
use crate::plans::{ExprIR, LiteralValue};
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

#[recursive::recursive]
fn aexpr_to_skip_batch_predicate_rec(
    e: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
    depth: usize,
) -> Option<Node> {
    use Operator as O;

    macro_rules! rec {
        ($node:expr) => {{
            aexpr_to_skip_batch_predicate_rec($node, expr_arena, schema, depth + 1)
        }};
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
        ($op:ident, $l:expr, $r:expr) => {{
            expr_arena.add(AExpr::BinaryExpr {
                left: $l,
                op: O::$op,
                right: $r,
            })
        }};
    }
    macro_rules! lt {
        ($l:expr, $r:expr) => {{
            binexpr!(Lt, $l, $r)
        }};
    }
    macro_rules! gt {
        ($l:expr, $r:expr) => {{
            binexpr!(Gt, $l, $r)
        }};
    }
    macro_rules! le {
        ($l:expr, $r:expr) => {{
            binexpr!(LtEq, $l, $r)
        }};
    }
    macro_rules! ge {
        ($l:expr, $r:expr) => {{
            binexpr!(GtEq, $l, $r)
        }};
    }
    macro_rules! eq_missing {
        ($l:expr, $r:expr) => {{
            binexpr!(EqValidity, $l, $r)
        }};
    }
    macro_rules! all {
        ($i:expr) => {{
            expr_arena.add(AExpr::Function {
                input: vec![ExprIR::new($i, OutputName::Alias(PlSmallStr::EMPTY))],
                function: FunctionExpr::Boolean(BooleanFunction::All { ignore_nulls: true }),
                options: FunctionOptions::default(),
            })
        }};
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
            eq_missing!(expr, idx_zero)
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
    macro_rules! col {
        (len) => {{
            col!(PlSmallStr::from_static("len"))
        }};
        ($name:expr) => {{
            expr_arena.add(AExpr::Column($name))
        }};
        (min: $name:expr) => {{
            col!(format_pl_smallstr!("{}_min", $name))
        }};
        (max: $name:expr) => {{
            col!(format_pl_smallstr!("{}_max", $name))
        }};
        (null_count: $name:expr) => {{
            col!(format_pl_smallstr!("{}_nc", $name))
        }};
    }
    macro_rules! lv {
        ($lv:expr) => {{
            expr_arena.add(AExpr::Literal(LiteralValue::OtherScalar(Scalar::from($lv))))
        }};
        (idx: $lv:expr) => {{
            expr_arena.add(AExpr::Literal(LiteralValue::new_idxsize($lv)))
        }};
        (bool: $lv:expr) => {{
            expr_arena.add(AExpr::Literal(LiteralValue::Boolean($lv)))
        }};
    }

    if let Some(Some(lv)) = constant_evaluate(e, expr_arena, schema, 0) {
        if let Some(av) = lv.to_any_value() {
            return match av {
                AnyValue::Null => Some(lv!(bool: false)),
                AnyValue::Boolean(b) => Some(lv!(bool: b)),
                _ => None,
            };
        }
    }

    match expr_arena.get(e) {
        AExpr::Explode(_) => None,
        AExpr::Alias(_, _) => None,
        AExpr::Column(_) => None,
        AExpr::Literal(_) => None,
        AExpr::BinaryExpr { left, op, right } => {
            let left = *left;
            let right = *right;

            match op {
                O::Eq | O::EqValidity => {
                    let ((col, _), (_, lv_node)) =
                        get_binary_expr_col_and_lv(left, right, expr_arena, schema)?;

                    let op = *op;
                    let col = col.clone();

                    // col(A) == B --> min(A) > B || max(A) < B
                    let col_min = col!(min: col);
                    let col_max = col!(max: col);

                    let dtype = schema.get(&col)?;
                    let min_is_defined = is_stat_defined!(col_min, dtype);
                    let max_is_defined = is_stat_defined!(col_max, dtype);

                    let min_gt = gt!(col_min, lv_node);
                    let min_gt = and!(min_is_defined, min_gt);

                    let max_lt = lt!(col_max, lv_node);
                    let max_lt = and!(max_is_defined, max_lt);

                    let non_null_case = or!(min_gt, max_lt);
                    let lv_has_nulls = has_nulls!(lv_node);

                    Some(if matches!(op, O::Eq) {
                        or!(lv_has_nulls, non_null_case)
                    } else {
                        // col(A) == null --> NULL_COUNT(A) == 0
                        let col_nc = col!(null_count: col);
                        let idx_zero = lv!(idx: 0);
                        let t = eq_missing!(col_nc, idx_zero);
                        let null_case = and!(lv_has_nulls, t);

                        let lv_no_nulls = has_no_nulls!(lv_node);
                        let non_null_case = and!(lv_no_nulls, non_null_case);
                        or!(null_case, non_null_case)
                    })
                },
                O::NotEq | O::NotEqValidity => {
                    let ((col, _), (_, lv_node)) =
                        get_binary_expr_col_and_lv(left, right, expr_arena, schema)?;

                    let op = *op;
                    let col = col.clone();

                    let col_min = col!(min: col);
                    let col_max = col!(max: col);
                    let min_eq = eq_missing!(col_min, lv_node);
                    let max_eq = eq_missing!(col_max, lv_node);
                    let non_null_case = and!(min_eq, max_eq);

                    // col(A) !=     null --> false
                    // col(A) !=~    null --> NULL_COUNT(A) == LEN
                    // col(A) !=(~)  B    --> min(A) == B && max(A) == B

                    let lv_no_nulls = has_no_nulls!(lv_node);

                    Some(if matches!(op, O::NotEq) {
                        and!(lv_no_nulls, non_null_case)
                    } else {
                        let col_nc = col!(null_count: col);
                        let len = col!(len);
                        let t = eq_missing!(col_nc, len);
                        let null_case = or!(lv_no_nulls, t);

                        let lv_has_nulls = has_nulls!(lv_node);
                        let non_null_case = or!(lv_has_nulls, non_null_case);
                        and!(null_case, non_null_case)
                    })
                },
                O::Lt | O::GtEq => {
                    let ((col, col_node), (_, lv_node)) =
                        get_binary_expr_col_and_lv(left, right, expr_arena, schema)?;

                    let is_col_less_than_lv = matches!(op, O::Lt) == (col_node == left);

                    let col = col.clone();
                    let dtype = schema.get(&col)?;
                    let expr = if is_col_less_than_lv {
                        // col(A) < B --> min(A) >= B
                        let col_min = col!(min: col);
                        let min_is_defined = is_stat_defined!(col_min, dtype);
                        let min_ge = ge!(col_min, lv_node);
                        and!(min_is_defined, min_ge)
                    } else {
                        // col(A) >= B --> max(A) < B
                        let col_max = col!(max: col);
                        let max_is_defined = is_stat_defined!(col_max, dtype);
                        let max_lt = lt!(col_max, lv_node);
                        and!(max_is_defined, max_lt)
                    };

                    let has_nulls = has_nulls!(lv_node);
                    Some(or!(has_nulls, expr))
                },
                O::Gt | O::LtEq => {
                    let ((col, col_node), (_, lv_node)) =
                        get_binary_expr_col_and_lv(left, right, expr_arena, schema)?;

                    let is_col_greater_than_lv = matches!(op, O::Gt) == (col_node == left);

                    let col = col.clone();
                    let dtype = schema.get(&col)?;
                    let expr = if is_col_greater_than_lv {
                        // col(A) > B --> max(A) <= B
                        let col_max = col!(max: col);
                        let max_is_defined = is_stat_defined!(col_max, dtype);
                        let max_le = le!(col_max, lv_node);
                        and!(max_is_defined, max_le)
                    } else {
                        // col(A) <= B --> min(A) > B
                        let col_min = col!(min: col);
                        let min_is_defined = is_stat_defined!(col_min, dtype);
                        let min_gt = gt!(col_min, lv_node);
                        and!(min_is_defined, min_gt)
                    };

                    let has_nulls = has_nulls!(lv_node);
                    Some(or!(has_nulls, expr))
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
                BooleanFunction::IsIn => {
                    let lv_node = input[1].node();
                    match (
                        into_column(input[0].node(), expr_arena, schema, 0),
                        constant_evaluate(lv_node, expr_arena, schema, 0),
                    ) {
                        (Some(col), Some(_)) => {
                            // col(A).is_in([B1, ..., Bn]) ->
                            //     (s.null_count() == 0 || (nc(A) == 0))
                            //  && (
                            //      all(min(A) > [B1, ..., Bn)]) ||
                            //      all(max(A) < [B1, ..., Bn)])
                            //  )
                            let col = col.clone();
                            let dtype = schema.get(&col)?;

                            let col_min = col!(min: col);
                            let col_max = col!(max: col);

                            let min_is_defined = is_stat_defined!(col_min, dtype);
                            let max_is_defined = is_stat_defined!(col_max, dtype);

                            let min_gt = gt!(col_min, lv_node);
                            let min_gt = all!(min_gt);
                            let min_gt = and!(min_is_defined, min_gt);

                            let max_lt = lt!(col_max, lv_node);
                            let max_lt = all!(max_lt);
                            let max_lt = and!(max_is_defined, max_lt);

                            let expr = or!(min_gt, max_lt);

                            let col_nc = col!(null_count: col);
                            let idx_zero = lv!(idx: 0);
                            let col_has_no_nulls = eq_missing!(col_nc, idx_zero);

                            let lv_has_not_nulls = has_no_nulls!(lv_node);
                            let null_case = or!(lv_has_not_nulls, col_has_no_nulls);

                            Some(and!(null_case, expr))
                        },
                        _ => None,
                    }
                },
                BooleanFunction::IsNull => {
                    let col = into_column(input[0].node(), expr_arena, schema, 0)?;

                    // col(A).is_null() --> NULL_COUNT(A) == 0
                    let col_nc = col!(null_count: col);
                    let idx_zero = lv!(idx: 0);
                    Some(eq_missing!(col_nc, idx_zero))
                },
                BooleanFunction::IsNotNull => {
                    let col = into_column(input[0].node(), expr_arena, schema, 0)?;

                    // col(A).is_not_null() --> NULL_COUNT(A) == LEN
                    let col_nc = col!(null_count: col);
                    let len = col!(len);
                    Some(eq_missing!(col_nc, len))
                },
                #[cfg(feature = "is_between")]
                BooleanFunction::IsBetween { closed } => {
                    let col = into_column(input[0].node(), expr_arena, schema, 0)?;

                    let left_node = input[1].node();
                    let right_node = input[2].node();

                    _ = constant_evaluate(left_node, expr_arena, schema, 0)?;
                    _ = constant_evaluate(right_node, expr_arena, schema, 0)?;

                    let col = col.clone();
                    let closed = *closed;
                    let dtype = schema.get(&col)?;

                    let lhs_no_nulls = has_no_nulls!(left_node);
                    let rhs_no_nulls = has_no_nulls!(right_node);

                    let col_min = col!(min: col);
                    let col_max = col!(max: col);

                    use polars_ops::series::ClosedInterval;
                    let (left, right) = match closed {
                        ClosedInterval::Both => (lt!(col_max, left_node), gt!(col_min, right_node)),
                        ClosedInterval::Left => (lt!(col_max, left_node), ge!(col_min, right_node)),
                        ClosedInterval::Right => {
                            (le!(col_max, left_node), gt!(col_min, right_node))
                        },
                        ClosedInterval::None => (le!(col_max, left_node), ge!(col_min, right_node)),
                    };

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
}
