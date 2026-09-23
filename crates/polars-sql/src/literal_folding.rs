//! Exact numeric literals and exact folding of arithmetic between them.
//!
//! An `<exact numeric literal>` (digits with an optional `.`, no exponent) is a decimal
//! with the spelled scale, so `1.0000` is `Decimal(5, 4)`. Arithmetic between literals
//! containing a decimal point is folded with the same kernels and result scales as
//! decimal column arithmetic, and raises where that would. Integer-only arithmetic is
//! left to the engine.
//!
//! An expression of literals only, such as `DATE '1994-01-01' + INTERVAL '1' YEAR`, is
//! evaluated once by the engine, so it is a plain literal in the plan.

use polars_compute::decimal::{
    DEC128_MAX_PREC, dec128_add_scaled, dec128_mul_scaled, dec128_sub_scaled,
};
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::prelude::{Expr, Literal};
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Expr as SQLExpr, UnaryOperator as SQLUnaryOperator,
    Value as SQLValue, ValueWithSpan,
};

/// A decimal as `(mantissa, precision, scale)`: `mantissa / 10^scale`.
pub(crate) type ExactLiteral = (i128, usize, usize);

/// The smallest precision holding `mantissa` at `scale`.
fn precision_of(mantissa: i128, scale: usize) -> usize {
    let digits = mantissa
        .unsigned_abs()
        .checked_ilog10()
        .map_or(1, |d| d as usize + 1);
    digits.max(scale).max(1)
}

/// Parse an `<exact numeric literal>` spelling without rounding; `None` if it isn't one,
/// or it needs more than 38 digits (the caller then raises).
pub(crate) fn parse_exact_literal(s: &str) -> Option<ExactLiteral> {
    let (int_part, frac_part) = s.split_once('.').unwrap_or((s, ""));
    if int_part.is_empty() && frac_part.is_empty()
        || !int_part
            .bytes()
            .chain(frac_part.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let scale = frac_part.len();
    if int_part.trim_start_matches('0').len() + scale > DEC128_MAX_PREC {
        return None;
    }
    let mantissa = format!("{int_part}{frac_part}").parse().ok()?;
    Some((mantissa, precision_of(mantissa, scale), scale))
}

pub(crate) fn decimal_lit((mantissa, precision, scale): ExactLiteral) -> Expr {
    Scalar::new_decimal(mantissa, precision, scale).lit()
}

/// Whether `expr` is literal arithmetic containing a literal with a decimal point.
fn has_dotted_literal(expr: &SQLExpr) -> bool {
    match expr {
        SQLExpr::Value(ValueWithSpan {
            value: SQLValue::Number(s, _),
            ..
        }) => s.contains('.'),
        SQLExpr::Nested(e) | SQLExpr::UnaryOp { expr: e, .. } => has_dotted_literal(e),
        SQLExpr::BinaryOp { left, right, .. } => {
            has_dotted_literal(left) || has_dotted_literal(right)
        },
        _ => false,
    }
}

/// `(l, sl, r, sr, s_out)`: the result of `l` (scale `sl`) with `r` (scale `sr`) at `s_out`.
type ScaledKernel = fn(i128, usize, i128, usize, usize) -> Option<i128>;

/// `Ok(None)` if `expr` isn't foldable literal arithmetic.
fn eval(expr: &SQLExpr) -> PolarsResult<Option<(i128, usize)>> {
    Ok(match expr {
        SQLExpr::Value(ValueWithSpan {
            value: SQLValue::Number(s, _),
            ..
        }) => parse_exact_literal(s).map(|(m, _, s)| (m, s)),
        SQLExpr::Nested(e) => eval(e)?,
        SQLExpr::UnaryOp { op, expr } => match op {
            SQLUnaryOperator::Plus => eval(expr)?,
            // Negating a Decimal128 mantissa can't overflow.
            SQLUnaryOperator::Minus => eval(expr)?.map(|(m, s)| (-m, s)),
            _ => None,
        },
        SQLExpr::BinaryOp { left, op, right } => combine(left, op, right)?,
        _ => None,
    })
}

fn combine(
    left: &SQLExpr,
    op: &SQLBinaryOperator,
    right: &SQLExpr,
) -> PolarsResult<Option<(i128, usize)>> {
    let (f, name): (ScaledKernel, _) = match op {
        SQLBinaryOperator::Plus => (dec128_add_scaled, "addition"),
        SQLBinaryOperator::Minus => (dec128_sub_scaled, "subtraction"),
        SQLBinaryOperator::Multiply => (dec128_mul_scaled, "multiplication"),
        _ => return Ok(None),
    };
    let (Some((l, sl)), Some((r, sr))) = (eval(left)?, eval(right)?) else {
        return Ok(None);
    };
    let scale = match op {
        SQLBinaryOperator::Multiply => {
            let scale = sl + sr;
            polars_ensure!(
                scale <= DEC128_MAX_PREC,
                SQLInterface: "numeric value out of range: multiplication result scale {} exceeds {}",
                scale, DEC128_MAX_PREC
            );
            scale
        },
        _ => sl.max(sr),
    };
    let value = f(l, sl, r, sr, scale).ok_or_else(|| {
        polars_err!(
            SQLInterface: "numeric value out of range: {} result doesn't fit Decimal({}, {})",
            name, DEC128_MAX_PREC, scale
        )
    })?;
    Ok(Some((value, scale)))
}

/// Fold `left <op> right` when both sides are numeric literal arithmetic and at least one
/// literal has a decimal point; `Ok(None)` leaves the expression to ordinary translation.
pub(crate) fn try_fold_decimal_arithmetic(
    left: &SQLExpr,
    op: &SQLBinaryOperator,
    right: &SQLExpr,
) -> PolarsResult<Option<Expr>> {
    if !(has_dotted_literal(left) || has_dotted_literal(right)) {
        return Ok(None);
    }
    Ok(combine(left, op, right)?.map(|(m, s)| decimal_lit((m, precision_of(m, s), s))))
}

/// Evaluate `expr`, built from literals only, into one literal.
pub(crate) fn fold_scalar(expr: Expr) -> PolarsResult<Expr> {
    let df = DataFrame::empty().lazy().select([expr]).collect()?;
    let value = df.columns()[0].get(0)?.into_static();
    Ok(lit(Scalar::new(df.columns()[0].dtype().clone(), value)))
}
