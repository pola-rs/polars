//! Small helpers shared across translation.

use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_plan::dsl::Expr;

/// The error for an expression that the JIT does not translate.
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

/// The location of a part of the expression at `location`, used in error messages.
pub(super) fn at(location: &str, part: &str) -> String {
    format!("{location}.{part}")
}
