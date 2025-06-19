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
            let left = expr_arena.get(left);
            let right = expr_arena.get(right);

            // true AND x => x
            if matches!(
                left,
                AExpr::Literal(lv) if lv.bool() == Some(true)
            ) && in_filter
            {
                // Only in filter as we might change the name from "literal"
                // to whatever lhs columns is.
                return Some(right.clone());
            }

            // x AND true => x
            if matches!(
                right,
                AExpr::Literal(lv) if lv.bool() == Some(true)
            ) {
                return Some(left.clone());
            }

            // x AND false -> false
            // FIXME: we need an optimizer redesign to allow x & false to be optimized
            // in general as we can forget the length of a series otherwise.
            // In filter we allow it as the length is not important there.
            if (matches!(left, AExpr::Literal(_)) | in_filter)
                && matches!(
                    right,
                    AExpr::Literal(lv) if lv.bool() == Some(false)
                )
            {
                return Some(AExpr::Literal(Scalar::from(false).into()));
            }
        },
        O::Or => {
            let left = expr_arena.get(left);
            let right = expr_arena.get(right);

            // false OR x => x
            if matches!(
                left,
                AExpr::Literal(lv) if lv.bool() == Some(false)
            ) && in_filter
            {
                // Only in filter as we might change the name from "literal"
                // to whatever lhs columns is.
                return Some(right.clone());
            }

            // x OR false => x
            if matches!(
                right,
                AExpr::Literal(lv) if lv.bool() == Some(false)
            ) {
                return Some(left.clone());
            }
            // true OR x => true
            // FIXME: we need an optimizer redesign to allow true | x to be optimized
            // in general as we can forget the length of a series otherwise.
            // In filter we allow it as the length is not important there.
            if (matches!(left, AExpr::Literal(_)) | in_filter)
                && matches!(
                    right,
                    AExpr::Literal(lv) if lv.bool() == Some(true)
                )
            {
                return Some(AExpr::Literal(Scalar::from(true).into()));
            }

            // x OR true => true
            // FIXME: we need an optimizer redesign to allow true | x to be optimized
            // in general as we can forget the length of a series otherwise.
            // In filter we allow it as the length is not important there.
            if matches!(
                left,
                    AExpr::Literal(lv) if lv.bool() == Some(true)
            ) && (matches!(right, AExpr::Literal(_)) | in_filter)
            {
                return Some(AExpr::Literal(Scalar::from(true).into()));
            }
        },

        _ => {},
    }
    None
}
