//! Translation of binary operators.

use polars_error::PolarsResult;
use polars_plan::dsl::{Expr, Operator};

use super::Translator;
use crate::jit_program::{BinaryOp, JitProgram};
use crate::translate::helpers::at;

impl Translator<'_> {
    /// `+`, `-`, and `*`, with the result type Polars gives the expression.
    pub(super) fn arithmetic(
        &self,
        expression: &Expr,
        left: &Expr,
        op: Operator,
        right: &Expr,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let op = match op {
            Operator::Plus => BinaryOp::Add,
            Operator::Minus => BinaryOp::Subtract,
            Operator::Multiply => BinaryOp::Multiply,
            _ => unreachable!("`expression` only sends arithmetic operators here"),
        };
        Ok(JitProgram::Binary {
            op,
            left: Box::new(self.expression(left, &at(location, "left"))?),
            right: Box::new(self.expression(right, &at(location, "right"))?),
            dtype: expression.to_field(self.schema)?.dtype,
        })
    }
}
