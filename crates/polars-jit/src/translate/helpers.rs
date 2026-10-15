//! Small helpers shared across translation.

use polars_error::{PolarsError, polars_err};
use polars_plan::dsl::Expr;

/// The error for an expression that the JIT does not translate.
pub(super) fn unsupported(expression: &Expr, location: &str) -> PolarsError {
    polars_err!(InvalidOperation:
        "expression is not supported by the JIT at {location}: {expression:?}")
}

/// The location of a part of the expression at `location`, used in error messages.
pub(super) fn at(location: &str, part: &str) -> String {
    format!("{location}.{part}")
}
