//! Small helpers shared across translation.

use polars_core::prelude::AnyValue;
use polars_core::utils::try_get_supertype;
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_plan::dsl::{Expr, Operator};
use polars_plan::prelude::LiteralValue;

use crate::jit_program::{JitProgram, ScalarType};

/// The error for an expression, or a form of one, that the JIT does not translate.
pub(super) fn unsupported(expression: &Expr, location: &str) -> PolarsError {
    polars_err!(InvalidOperation:
        "expression is not supported by the JIT at {location}: {expression:?}")
}

/// The inputs of a function that takes exactly `N` of them. Any other number of inputs is a
/// form of the function that foldv does not support.
pub(super) fn inputs<'e, const N: usize>(
    input: &'e [Expr],
    expression: &Expr,
    location: &str,
) -> PolarsResult<&'e [Expr; N]> {
    input
        .try_into()
        .map_err(|_| unsupported(expression, location))
}

/// Whether `expression` is an untyped null, which takes the type of what it is combined with.
pub(super) fn is_null_literal(expression: &Expr) -> bool {
    matches!(expression, Expr::Literal(LiteralValue::Scalar(value))
        if matches!(value.as_any_value(), AnyValue::Null) && value.dtype().is_null())
}

/// The supertype Polars gives two types, such as Float64 for Int32 and Float64.
pub(super) fn common_type(left: ScalarType, right: ScalarType) -> PolarsResult<ScalarType> {
    ScalarType::from_dtype(&try_get_supertype(&left.dtype(), &right.dtype())?)
}

/// Integer division and modulo use the same numeric coercion as addition. Polars' schema
/// inference for these operators can incorrectly adopt an unsigned column's type when a
/// negative untyped literal is on the left; its execution instead coerces both to Int64.
/// True division also needs column-first inference for untyped literals and nulls. These
/// changes apply only while asking for dtypes, leaving the actual operations unchanged.
pub(super) fn with_arithmetic_types(expression: &Expr) -> Expr {
    expression.clone().map_expr(|expr| match expr {
        Expr::BinaryExpr {
            left,
            op: Operator::TrueDivide,
            right,
        } if matches!(&*left, Expr::Literal(_)) && !matches!(&*right, Expr::Literal(_)) => {
            Expr::BinaryExpr {
                left: right,
                op: Operator::TrueDivide,
                right: left,
            }
        },
        Expr::BinaryExpr {
            left,
            op: Operator::FloorDivide | Operator::Modulus | Operator::RustDivide,
            right,
        } => Expr::BinaryExpr {
            left,
            op: Operator::Plus,
            right,
        },
        expr => expr,
    })
}

/// `program`, negated when `negate` is set; for function pairs such as `is_null`/`is_not_null`.
pub(super) fn negated_if(negate: bool, program: JitProgram) -> JitProgram {
    if negate {
        JitProgram::Not(Box::new(program))
    } else {
        program
    }
}

/// The location of a part of the expression at `location`, used in error messages.
pub(super) fn at(location: &str, part: &str) -> String {
    format!("{location}.{part}")
}
