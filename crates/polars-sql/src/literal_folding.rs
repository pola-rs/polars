//! Exact folding of arithmetic between numeric SQL literals.
//!
//! `.06 + 0.01` is computed on the literal spellings as fixed-point values (exact within
//! `i128`) and converted once to the ordinary `Float64` literal. Integer-only arithmetic
//! is left to the engine.

use polars_compute::decimal::exact;
use polars_plan::prelude::{Expr, lit};
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Expr as SQLExpr, UnaryOperator as SQLUnaryOperator,
    Value as SQLValue, ValueWithSpan,
};

/// A literal as `(mantissa, scale)`: `mantissa / 10^scale`.
type Fixed = (i128, usize);

/// Read a numeric literal spelling (digits with an optional `.`) without rounding.
fn parse_spelling(s: &str) -> Option<Fixed> {
    let (int_part, frac_part) = s.split_once('.').unwrap_or((s, ""));
    if !int_part
        .bytes()
        .chain(frac_part.bytes())
        .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let mantissa = format!("{int_part}{frac_part}").parse().ok()?;
    Some((mantissa, frac_part.len()))
}

fn eval(expr: &SQLExpr) -> Option<Fixed> {
    match expr {
        SQLExpr::Value(ValueWithSpan {
            value: SQLValue::Number(s, _),
            ..
        }) => parse_spelling(s),
        SQLExpr::Nested(e) => eval(e),
        SQLExpr::UnaryOp { op, expr } => match op {
            SQLUnaryOperator::Plus => eval(expr),
            SQLUnaryOperator::Minus => exact::neg(eval(expr)?),
            _ => None,
        },
        SQLExpr::BinaryOp { left, op, right } => combine(left, op, right),
        _ => None,
    }
}

fn combine(left: &SQLExpr, op: &SQLBinaryOperator, right: &SQLExpr) -> Option<Fixed> {
    let f = match op {
        SQLBinaryOperator::Plus => exact::add,
        SQLBinaryOperator::Minus => exact::sub,
        SQLBinaryOperator::Multiply => exact::mul,
        _ => return None,
    };
    f(eval(left)?, eval(right)?)
}

/// One correctly rounded conversion of the exact result.
fn to_f64((mantissa, scale): Fixed) -> f64 {
    format!("{mantissa}e-{scale}").parse().unwrap()
}

/// Fold `left <op> right` when both sides are numeric literal arithmetic with a
/// decimal point; `None` leaves the expression to ordinary translation.
pub(crate) fn try_fold_decimal_arithmetic(
    left: &SQLExpr,
    op: &SQLBinaryOperator,
    right: &SQLExpr,
) -> Option<Expr> {
    let value = combine(left, op, right)?;
    (value.1 > 0).then(|| lit(to_f64(value)))
}
