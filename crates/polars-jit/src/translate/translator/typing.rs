//! Typing rules for subexpressions.

use polars_core::prelude::DataType;
use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::Expr;

use super::Translator;
use crate::jit_program::JitProgram;

impl Translator<'_> {
    /// Translate an expression that must produce a Boolean, such as a predicate.
    pub(super) fn boolean(&self, expression: &Expr, location: &str) -> PolarsResult<JitProgram> {
        let program = self.expression(expression, location)?;
        polars_ensure!(program.dtype() == &DataType::Boolean, InvalidOperation:
            "predicate at {location} must produce Boolean");
        Ok(program)
    }
}
