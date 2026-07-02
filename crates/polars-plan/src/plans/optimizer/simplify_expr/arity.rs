use super::*;
use crate::dsl::Operator;

pub(super) fn simplify_binary(
    left: Node,
    op: Operator,
    right: Node,
    ctx: OptimizeExprContext,
    maintain_errors: bool,
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

            if in_filter {
                // `A AND NOT(A)` is always false (either operand may be the
                // `NOT`). Catches cases the earlier rewrites left alone, like
                // `NOT(col)` on a bool column or `NOT(is_in(col, values))`.
                if is_self_negation(left, left_ae, right_ae, expr_arena, maintain_errors)
                    || is_self_negation(right, right_ae, left_ae, expr_arena, maintain_errors)
                {
                    return Some(AExpr::Literal(Scalar::from(false).into()));
                }

                // Two comparisons on the same operands that can never both hold,
                // e.g. `(col > N) AND (col <= N)` or `(col > N) AND (col < N)`.
                if let (
                    AExpr::BinaryExpr {
                        left: l1,
                        op: op1,
                        right: r1,
                    },
                    AExpr::BinaryExpr {
                        left: l2,
                        op: op2,
                        right: r2,
                    },
                ) = (left_ae, right_ae)
                {
                    if comparisons_contradict(*op1, *op2) {
                        let l1_ae = expr_arena.get(*l1);
                        let l2_ae = expr_arena.get(*l2);
                        let r1_ae = expr_arena.get(*r1);
                        let r2_ae = expr_arena.get(*r2);
                        if l1_ae.is_expr_equal_to(l2_ae, expr_arena)
                            && r1_ae.is_expr_equal_to(r2_ae, expr_arena)
                            && is_safe_to_drop(*l1, expr_arena, maintain_errors)
                            && is_safe_to_drop(*r1, expr_arena, maintain_errors)
                        {
                            return Some(AExpr::Literal(Scalar::from(false).into()));
                        }
                    }
                }
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

// Returns the inner expression if `ae` is `NOT(inner)`, else `None`.
fn is_not_of(ae: &AExpr) -> Option<Node> {
    if let AExpr::Function {
        input,
        function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
        ..
    } = ae
    {
        if input.len() == 1 {
            return Some(input[0].node());
        }
    }
    None
}

// True when `a AND b` is `A AND NOT(A)` (i.e. `b` is `NOT(a)`) and `a` is safe
// to drop. Checking only `a` suffices: `b` is `NOT(a)`, which wraps the same
// expression, so it is droppable whenever `a` is.
fn is_self_negation(
    a: Node,
    a_ae: &AExpr,
    b_ae: &AExpr,
    expr_arena: &Arena<AExpr>,
    maintain_errors: bool,
) -> bool {
    let Some(inner) = is_not_of(b_ae) else {
        return false;
    };
    a_ae.is_expr_equal_to(expr_arena.get(inner), expr_arena)
        && is_safe_to_drop(a, expr_arena, maintain_errors)
}

// Bitset positions for the three outcomes of comparing two values.
const CMP_LT: u8 = 1; // x < y
const CMP_EQ: u8 = 2; // x == y
const CMP_GT: u8 = 4; // x > y

// The cases a comparison operator is true in, as a bitset over `<` / `==` / `>`.
// `None` for non-comparison operators.
fn comparison_cases(op: Operator) -> Option<u8> {
    use Operator::*;
    Some(match op {
        Lt => CMP_LT,
        LtEq => CMP_LT | CMP_EQ,
        Gt => CMP_GT,
        GtEq => CMP_GT | CMP_EQ,
        Eq => CMP_EQ,
        NotEq => CMP_LT | CMP_GT,
        EqValidity | NotEqValidity | And | Or | Xor | LogicalAnd | LogicalOr | Plus | Minus
        | Multiply | RustDivide | TrueDivide | FloorDivide | Modulus => return None,
    })
}

// Whether `(x op1 y) AND (x op2 y)` can never both hold: the operators share no
// case (their true-sets over `<` / `==` / `>` are disjoint). `false` if either
// is not a comparison.
fn comparisons_contradict(op1: Operator, op2: Operator) -> bool {
    match (comparison_cases(op1), comparison_cases(op2)) {
        (Some(a), Some(b)) => a & b == 0,
        _ => false,
    }
}

// Whether we can drop this expression when collapsing a contradiction to
// `false`. A random (non-deterministic) expression is never safe to drop: two
// copies of it aren't the same value, so they don't really contradict. An
// expression that can error is safe to drop only when the user has not asked to
// keep errors (same rule as predicate pushdown).
fn is_safe_to_drop(node: Node, expr_arena: &Arena<AExpr>, maintain_errors: bool) -> bool {
    if is_inherently_nondeterministic(node, expr_arena) {
        return false;
    }
    let ae = expr_arena.get(node);
    let mut group = ExprPushdownGroup::Pushable;
    group.update_with_expr_rec(ae, expr_arena, None);
    !group.blocks_pushdown(maintain_errors)
}
