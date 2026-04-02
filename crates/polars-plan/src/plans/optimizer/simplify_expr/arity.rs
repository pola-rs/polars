use super::*;
use crate::dsl::Operator;

pub(super) fn simplify_binary(
    left: Node,
    op: Operator,
    right: Node,
    ctx: OptimizeExprContext,
    expr_arena: &mut Arena<AExpr>,
) -> Option<AExpr> {
    let in_filter = ctx.in_filter;

    use Operator as O;
    match op {
        O::And => {
            let left_ae = expr_arena.get(left);
            let right_ae = expr_arena.get(right);

            // true AND x => x
            if matches!(
                left_ae,
                AExpr::Literal(lv) if lv.bool() == Some(true)
            ) && in_filter
            {
                // Only in filter as we might change the name from "literal"
                // to whatever lhs columns is.
                return Some(right_ae.clone());
            }

            // x AND true => x
            if matches!(
                right_ae,
                AExpr::Literal(lv) if lv.bool() == Some(true)
            ) {
                return Some(left_ae.clone());
            }

            // x AND false -> false
            // TODO: we need an optimizer redesign to allow x & false to be optimized
            // in general as we can forget the length of a series otherwise.
            // In filter we allow it as the length is not important there.
            if (is_scalar_ae(left, expr_arena) | in_filter)
                && matches!(
                    right_ae,
                    AExpr::Literal(lv) if lv.bool() == Some(false)
                )
            {
                return Some(AExpr::Literal(Scalar::from(false).into()));
            }
        },
        O::Or => {
            let left_ae = expr_arena.get(left);
            let right_ae = expr_arena.get(right);

            // false OR x => x
            if matches!(
                left_ae,
                AExpr::Literal(lv) if lv.bool() == Some(false)
            ) && in_filter
            {
                // Only in filter as we might change the name from "literal"
                // to whatever lhs columns is.
                return Some(right_ae.clone());
            }

            // x OR false => x
            if matches!(
                right_ae,
                AExpr::Literal(lv) if lv.bool() == Some(false)
            ) {
                return Some(left_ae.clone());
            }
            // true OR x => true
            // TODO: we need an optimizer redesign to allow true | x to be optimized
            // in general as we can forget the length of a series otherwise.
            // In filter we allow it as the length is not important there.
            if (is_scalar_ae(left, expr_arena) | in_filter)
                && matches!(
                    right_ae,
                    AExpr::Literal(lv) if lv.bool() == Some(true)
                )
            {
                return Some(AExpr::Literal(Scalar::from(true).into()));
            }

            // x OR true => true
            // TODO: we need an optimizer redesign to allow true | x to be optimized
            // in general as we can forget the length of a series otherwise.
            // In filter we allow it as the length is not important there.
            if matches!(
                left_ae,
                    AExpr::Literal(lv) if lv.bool() == Some(true)
            ) && (is_scalar_ae(right, expr_arena) | in_filter)
            {
                return Some(AExpr::Literal(Scalar::from(true).into()));
            }
        },

        _ => {},
    }
    None
}

pub(super) fn simplify_ternary(
    predicate: Node,
    truthy: Node,
    falsy: Node,
    expr_arena: &mut Arena<AExpr>,
) -> Option<AExpr> {
    let predicate = expr_arena.get(predicate);

    if let AExpr::Literal(lv) = predicate {
        match lv.bool() {
            None => {},
            Some(true) => {
                // Only replace if both are scalar or both are not scalar and are the same length,
                // the latter is tested by checking if they are elementwise.
                let t_is_scalar = is_scalar_ae(truthy, expr_arena);
                let f_is_scalar = is_scalar_ae(falsy, expr_arena);

                if t_is_scalar == f_is_scalar
                    && is_elementwise_rec(truthy, expr_arena)
                    && is_elementwise_rec(falsy, expr_arena)
                {
                    return Some(expr_arena.get(truthy).clone());
                }
            },
            Some(false) => {
                let t_is_scalar = is_scalar_ae(truthy, expr_arena);
                let f_is_scalar = is_scalar_ae(falsy, expr_arena);

                if t_is_scalar == f_is_scalar
                    && is_elementwise_rec(truthy, expr_arena)
                    && is_elementwise_rec(falsy, expr_arena)
                {
                    return Some(expr_arena.get(falsy).clone());
                }
            },
        }
    }

    None
}
