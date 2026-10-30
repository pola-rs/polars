use super::*;
use crate::dsl::Operator;

pub(super) fn simplify_binary(
    left: Node,
    op: Operator,
    right: Node,
    schema: &Schema,
    ctx: OptimizeExprContext,
    maintain_errors: bool,
    expr_arena: &mut Arena<AExpr>,
) -> Option<AExpr> {
    // These rewrites may turn `null` into `false` or a column into a scalar, see
    // `OptimizeExprContext::is_filter_predicate`.
    let is_filter_predicate = ctx.is_filter_predicate;

    use Operator as O;
    if let Some(out) =
        simplify_repeated_operand(left, op, right, schema, maintain_errors, expr_arena)
    {
        return Some(out);
    }

    match op {
        O::Eq | O::NotEq => return simplify_bool_literal_cmp(left, op, right, schema, expr_arena),
        O::And => {
            let left_ae = expr_arena.get(left);
            let right_ae = expr_arena.get(right);

            // true AND x => x
            if matches!(
                left_ae,
                AExpr::Literal(lv) if lv.bool() == Some(true)
            ) && (is_filter_predicate || is_boolean(right, schema, expr_arena))
            {
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
            if (is_scalar_ae(left, expr_arena) | is_filter_predicate)
                && matches!(
                    right_ae,
                    AExpr::Literal(lv) if lv.bool() == Some(false)
                )
            {
                return Some(AExpr::Literal(Scalar::from(false).into()));
            }

            if is_filter_predicate {
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
            ) && (is_filter_predicate || is_boolean(right, schema, expr_arena))
            {
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
            if (is_scalar_ae(left, expr_arena) | is_filter_predicate)
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
            ) && (is_scalar_ae(right, expr_arena) | is_filter_predicate)
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
    schema: &Schema,
    ctx: OptimizeExprContext,
    maintain_errors: bool,
    expr_arena: &mut Arena<AExpr>,
) -> Option<AExpr> {
    let predicate_ae = expr_arena.get(predicate);

    if let AExpr::Literal(lv) = predicate_ae
        && let Some(predicate) = lv.bool()
    {
        let (branch, other) = if predicate {
            (truthy, falsy)
        } else {
            (falsy, truthy)
        };
        // Replace if both are scalar or both are not scalar and are the same length, the
        // latter is tested by checking if they are elementwise. A scalar `other` is
        // broadcast to `branch`, so then any `branch` is fine.
        let b_is_scalar = is_scalar_ae(branch, expr_arena);
        let o_is_scalar = is_scalar_ae(other, expr_arena);

        if (b_is_scalar == o_is_scalar
            && (b_is_scalar
                || (is_elementwise_rec(branch, expr_arena)
                    && is_elementwise_rec(other, expr_arena))))
            || (!b_is_scalar && is_droppable_scalar(other, expr_arena))
        {
            return Some(expr_arena.get(branch).clone());
        }
        return None;
    }

    // when(p).then(x).otherwise(x) => x
    let truthy_ae = expr_arena.get(truthy);
    if truthy_ae.is_expr_equal_to(expr_arena.get(falsy), expr_arena) {
        return (!is_inherently_nondeterministic(truthy, expr_arena)
            && is_safe_to_drop(predicate, expr_arena, maintain_errors)
            && keeps_length(truthy, predicate, expr_arena)
            && is_boolean(predicate, schema, expr_arena))
        .then(|| truthy_ae.clone());
    }

    // when(p).then(true).otherwise(false) => p.fill_null(false)
    // when(p).then(false).otherwise(true) => p.fill_null(false).not()
    let negate = match (bool_lit(truthy, expr_arena), bool_lit(falsy, expr_arena)) {
        (Some(true), Some(false)) => false,
        (Some(false), Some(true)) => true,
        _ => return None,
    };
    // A predicate of another dtype must still raise when it runs.
    if !is_boolean(predicate, schema, expr_arena) {
        return None;
    }
    // A filter drops a `null` predicate like a `false` one.
    if ctx.is_filter_predicate && !negate {
        return Some(expr_arena.get(predicate).clone());
    }
    let false_lit = expr_arena.add(bool_literal(false));
    let out = function_of(
        &[predicate, false_lit],
        IRFunctionExpr::FillNull,
        expr_arena,
    );
    if !negate {
        return Some(out);
    }
    let out = expr_arena.add(out);
    Some(not(out, expr_arena))
}

fn function_of(input: &[Node], function: IRFunctionExpr, expr_arena: &Arena<AExpr>) -> AExpr {
    let input = input
        .iter()
        .map(|node| ExprIR::from_node(*node, expr_arena))
        .collect();
    let options = function.function_options();
    AExpr::Function {
        input,
        function,
        options,
    }
}

fn not(node: Node, expr_arena: &Arena<AExpr>) -> AExpr {
    function_of(
        &[node],
        IRFunctionExpr::Boolean(IRBooleanFunction::Not),
        expr_arena,
    )
}

fn bool_lit(node: Node, expr_arena: &Arena<AExpr>) -> Option<bool> {
    match expr_arena.get(node) {
        AExpr::Literal(lv) => lv.bool(),
        _ => None,
    }
}

// Whether an expression of `keep` and `drop` has the length of `keep`, so `drop` can go.
fn keeps_length(keep: Node, drop: Node, expr_arena: &Arena<AExpr>) -> bool {
    is_droppable_scalar(drop, expr_arena)
        || (is_length_preserving_ae(keep, expr_arena) && is_length_preserving_ae(drop, expr_arena))
}

// Whether `node` is a scalar that is broadcast to the other operands. The height of a struct
// evaluation does not count the columns its fields read, so it is never taken as scalar here.
fn is_droppable_scalar(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    if !is_scalar_ae(node, expr_arena) {
        return false;
    }
    #[cfg(feature = "dtype-struct")]
    if expr_arena
        .iter(node)
        .any(|(_, ae)| matches!(ae, AExpr::StructEval { .. }))
    {
        return false;
    }
    true
}

fn has_dtype(
    node: Node,
    schema: &Schema,
    expr_arena: &Arena<AExpr>,
    f: impl Fn(&DataType) -> bool,
) -> bool {
    let ctx = ToFieldContext::new(expr_arena, schema);
    expr_arena
        .get(node)
        .to_dtype(&ctx)
        .is_ok_and(|dtype| f(&dtype))
}

fn is_boolean(node: Node, schema: &Schema, expr_arena: &Arena<AExpr>) -> bool {
    has_dtype(node, schema, expr_arena, DataType::is_bool)
}

// `x == true` and `x != false` => `x`, `x == false` and `x != true` => `NOT(x)` for a
// boolean `x`. Both sides are `null` where `x` is `null`.
fn simplify_bool_literal_cmp(
    left: Node,
    op: Operator,
    right: Node,
    schema: &Schema,
    expr_arena: &mut Arena<AExpr>,
) -> Option<AExpr> {
    let (expr, value) = match (bool_lit(left, expr_arena), bool_lit(right, expr_arena)) {
        (None, Some(value)) => (left, value),
        (Some(value), None) => (right, value),
        _ => return None,
    };
    if !is_boolean(expr, schema, expr_arena) {
        return None;
    }
    if (op == Operator::Eq) == value {
        Some(expr_arena.get(expr).clone())
    } else {
        Some(not(expr, expr_arena))
    }
}

// `x & x` and `x | x` => `x`, also as bitwise operations on integers. For booleans also
// absorption: `x & (x | y)` and `x | (x & y)` => `x`, which holds under Kleene logic
// but not for integers, where a `null` in `y` gives `null`.
fn simplify_repeated_operand(
    left: Node,
    op: Operator,
    right: Node,
    schema: &Schema,
    maintain_errors: bool,
    expr_arena: &Arena<AExpr>,
) -> Option<AExpr> {
    let inner_op = match op {
        Operator::And => Operator::Or,
        Operator::Or => Operator::And,
        _ => return None,
    };

    let left_ae = expr_arena.get(left);
    if left_ae.is_expr_equal_to(expr_arena.get(right), expr_arena) {
        // Logical or bitwise `&` and `|` work on these dtypes.
        return (!is_inherently_nondeterministic(left, expr_arena)
            && has_dtype(left, schema, expr_arena, |dtype| {
                dtype.is_bool() || dtype.is_integer()
            }))
        .then(|| left_ae.clone());
    }

    for (keep, other) in [(left, right), (right, left)] {
        let AExpr::BinaryExpr {
            left: inner_left,
            op,
            right: inner_right,
        } = expr_arena.get(other)
        else {
            continue;
        };
        if *op != inner_op {
            continue;
        }
        let keep_ae = expr_arena.get(keep);
        let drop = if keep_ae.is_expr_equal_to(expr_arena.get(*inner_left), expr_arena) {
            *inner_right
        } else if keep_ae.is_expr_equal_to(expr_arena.get(*inner_right), expr_arena) {
            *inner_left
        } else {
            continue;
        };
        if !is_inherently_nondeterministic(keep, expr_arena)
            && is_safe_to_drop(drop, expr_arena, maintain_errors)
            && keeps_length(keep, drop, expr_arena)
            && is_boolean(keep, schema, expr_arena)
            && is_boolean(drop, schema, expr_arena)
        {
            return Some(keep_ae.clone());
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
        | Multiply | RustDivide | TrueDivide | FloorDivide | TruncDivide | Modulus => {
            return None;
        },
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

// `x.dt.year() == 2020` => `x.is_between(2020-01-01, 2021-01-01, closed="left")` and
// `x.dt.year() < 2020` => `x < 2020-01-01` for a Date or a Datetime without a time zone, so
// the comparison can use the statistics of `x`. Only for a filter: where `dt.year()` is
// null, the result is `false` instead of `null`.
#[cfg(all(feature = "temporal", feature = "is_between"))]
pub(super) fn simplify_year_comparison(
    left: Node,
    op: Operator,
    right: Node,
    schema: &Schema,
    expr_arena: &mut Arena<AExpr>,
) -> Option<AExpr> {
    use Operator as O;
    use chrono::{Datelike, NaiveDate};
    use polars_arrow::temporal_conversions::{
        EPOCH_DAYS_FROM_CE, MICROSECONDS_IN_DAY, MILLISECONDS_IN_DAY, NANOSECONDS_IN_DAY,
    };
    if !matches!(op, O::Lt | O::LtEq | O::Gt | O::GtEq | O::Eq | O::NotEq) {
        return None;
    }
    let (year_node, op, lit) = match (expr_arena.get(left), expr_arena.get(right)) {
        (_, AExpr::Literal(lv)) => (left, op, lv),
        (AExpr::Literal(lv), _) => (right, op.swap_operands()?, lv),
        _ => return None,
    };
    let AExpr::Function {
        input,
        function: IRFunctionExpr::TemporalExpr(IRTemporalFunction::Year),
        ..
    } = expr_arena.get(year_node)
    else {
        return None;
    };
    let x = input[0].node();
    // `!=` reads `x` twice, which is only cheap for a column.
    if op == O::NotEq && !matches!(expr_arena.get(x), AExpr::Column(_)) {
        return None;
    }
    let year = i32::try_from(lit.extract_i64().ok()?).ok()?;
    let dtype = expr_arena
        .get(x)
        .to_dtype(&ToFieldContext::new(expr_arena, schema))
        .ok()?;
    let per_day = match &dtype {
        DataType::Date => 1,
        DataType::Datetime(TimeUnit::Nanoseconds, None) => NANOSECONDS_IN_DAY,
        DataType::Datetime(TimeUnit::Microseconds, None) => MICROSECONDS_IN_DAY,
        DataType::Datetime(TimeUnit::Milliseconds, None) => MILLISECONDS_IN_DAY,
        _ => return None,
    };

    // The first or last value of `x` on `date`, if it fits.
    let date_value = |date: NaiveDate, last: bool| -> Option<Scalar> {
        let days = i64::from(date.num_days_from_ce()) - i64::from(EPOCH_DAYS_FROM_CE);
        let first = days.checked_mul(per_day)?;
        let value = if last {
            first.checked_add(per_day - 1)?
        } else {
            first
        };
        Some(match &dtype {
            DataType::Datetime(time_unit, _) => Scalar::new_datetime(value, *time_unit, None),
            _ => Scalar::new_date(i32::try_from(value).ok()?),
        })
    };
    let year_start = |year: i32| date_value(NaiveDate::from_ymd_opt(year, 1, 1)?, false);
    // The bounds on `x` for `from <= x.dt.year() < to`, where `None` is unbounded. `dt.year()`
    // is null outside the dates chrono supports, so an unbounded side stops there, unless `x`
    // can't hold values past it.
    let year_range = |from: Option<i32>, to: Option<i32>| -> Option<YearRange> {
        let low = match from {
            Some(year) => Some(year_start(year)?),
            None => date_value(NaiveDate::MIN, false),
        };
        let (high, closed) = match to {
            Some(year) => (Some(year_start(year)?), ClosedInterval::Left),
            None => (date_value(NaiveDate::MAX, true), ClosedInterval::Both),
        };
        Some(YearRange { low, high, closed })
    };

    let next_year = year.checked_add(1);
    if op == O::NotEq {
        let below = year_range(None, Some(year))?;
        let above = year_range(Some(next_year?), None)?;
        let left = below.into_expr(x, expr_arena)?;
        let right = above.into_expr(x, expr_arena)?;
        return Some(AExpr::BinaryExpr {
            left: expr_arena.add(left),
            op: O::Or,
            right: expr_arena.add(right),
        });
    }
    let (from, to) = match op {
        O::Lt => (None, Some(year)),
        O::LtEq => (None, Some(next_year?)),
        O::GtEq => (Some(year), None),
        O::Gt => (Some(next_year?), None),
        _ => (Some(year), Some(next_year?)),
    };
    year_range(from, to)?.into_expr(x, expr_arena)
}

#[cfg(all(feature = "temporal", feature = "is_between"))]
struct YearRange {
    low: Option<Scalar>,
    high: Option<Scalar>,
    closed: ClosedInterval,
}

#[cfg(all(feature = "temporal", feature = "is_between"))]
impl YearRange {
    fn into_expr(self, x: Node, expr_arena: &mut Arena<AExpr>) -> Option<AExpr> {
        let mut literal = |value: Scalar| expr_arena.add(AExpr::Literal(value.into()));
        Some(match (self.low, self.high) {
            (Some(low), Some(high)) => {
                let input = [x, literal(low), literal(high)];
                let function = IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween {
                    closed: self.closed,
                });
                function_of(&input, function, expr_arena)
            },
            (Some(low), None) => AExpr::BinaryExpr {
                left: x,
                op: Operator::GtEq,
                right: literal(low),
            },
            (None, Some(high)) => AExpr::BinaryExpr {
                left: x,
                op: if self.closed == ClosedInterval::Both {
                    Operator::LtEq
                } else {
                    Operator::Lt
                },
                right: literal(high),
            },
            (None, None) => return None,
        })
    }
}
