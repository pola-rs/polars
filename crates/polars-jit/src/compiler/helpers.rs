//! Small helpers shared across compilation.

use polars_error::{PolarsError, polars_err};

pub(super) fn compile_error(error: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "could not compile JIT program: {error}")
}
