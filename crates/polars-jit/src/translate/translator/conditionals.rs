//! Translation of `when(...).then(...).otherwise(...)`.

use polars_error::PolarsResult;
use polars_plan::dsl::Expr;

use super::Translator;
use crate::jit_program::JitProgram;
use crate::translate::helpers::at;

impl Translator<'_> {
    pub(super) fn ternary(
        &self,
        expression: &Expr,
        predicate: &Expr,
        truthy: &Expr,
        falsy: &Expr,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        Ok(JitProgram::Ternary {
            predicate: Box::new(self.boolean(predicate, &at(location, "predicate"))?),
            truthy: Box::new(self.expression(truthy, &at(location, "truthy"))?),
            falsy: Box::new(self.expression(falsy, &at(location, "falsy"))?),
            dtype: expression.to_field(self.schema)?.dtype,
        })
    }
}
