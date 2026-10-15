//! Translation of `when(...).then(...).otherwise(...)`.

use polars_error::PolarsResult;
use polars_plan::dsl::Expr;

use super::Translator;
use crate::jit_program::{JitProgram, ScalarType};
use crate::translate::helpers::{at, is_null_literal};

impl Translator<'_> {
    pub(super) fn ternary(
        &self,
        expression: &Expr,
        predicate: &Expr,
        truthy: &Expr,
        falsy: &Expr,
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        // A null literal takes its dtype from the other branch, so translate that first.
        let branches = self.translate_pair(
            (truthy, &at(location, "truthy")),
            (falsy, &at(location, "falsy")),
            expected,
            is_null_literal(truthy),
        )?;
        let dtype = self.result_type(expression, branches.0.dtype(), branches.1.dtype())?;
        let predicate = Box::new(self.boolean(predicate, &at(location, "predicate"))?);
        let (truthy, falsy) = self.cast_pair(branches, dtype)?;
        Ok(JitProgram::Ternary {
            predicate,
            truthy,
            falsy,
            dtype,
        })
    }
}
