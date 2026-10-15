//! Translation of Polars function expressions such as `not()`.
//!
//! Polars represents almost every method call as one `Expr::Function` node whose
//! `FunctionExpr` names the function; `Translator::expression` matches the function and calls
//! its implementation here.

use polars_error::PolarsResult;
use polars_plan::dsl::Expr;

use super::Translator;
use crate::jit_program::JitProgram;
use crate::translate::helpers::{at, inputs};

impl Translator<'_> {
    /// `not`.
    pub(super) fn not(
        &self,
        expression: &Expr,
        input: &[Expr],
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let [value] = inputs(input, expression, location)?;
        Ok(JitProgram::Not(Box::new(
            self.boolean(value, &at(location, "input"))?,
        )))
    }
}
