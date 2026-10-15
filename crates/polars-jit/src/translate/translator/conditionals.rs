//! Translation of `when(...).then(...).otherwise(...)`.

use polars_error::PolarsResult;
use polars_plan::dsl::Expr;

use super::Translator;
use crate::jit_program::{JitProgram, ScalarType};
use crate::translate::helpers::{at, common_type};

impl Translator<'_> {
    pub(super) fn ternary(
        &self,
        predicate: &Expr,
        truthy: &Expr,
        falsy: &Expr,
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let branches = self.translate_pair(
            (truthy, &at(location, "truthy")),
            (falsy, &at(location, "falsy")),
            expected,
            false,
        )?;
        let dtype = common_type(branches.0.dtype(), branches.1.dtype())?;
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
