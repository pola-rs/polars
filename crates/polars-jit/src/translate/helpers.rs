//! Small helpers shared across translation.

use polars_error::{PolarsError, polars_err};

/// The error for an expression that the JIT does not translate.
pub(super) fn unsupported() -> PolarsError {
    polars_err!(InvalidOperation: "foldv currently supports only `element + column`")
}
