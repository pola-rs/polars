//! Translation of binary operators.

use polars_error::PolarsResult;
use polars_plan::dsl::{Expr, Operator};

use super::Translator;
use crate::jit_program::{BinaryOp, BooleanOp, CompareOp, JitProgram};
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

    /// Comparisons of Int64 values.
    pub(super) fn comparison(
        &self,
        left: &Expr,
        op: Operator,
        right: &Expr,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let op = match op {
            Operator::Eq => CompareOp::Equal,
            Operator::NotEq => CompareOp::NotEqual,
            Operator::Lt => CompareOp::LessThan,
            Operator::LtEq => CompareOp::LessThanOrEqual,
            Operator::Gt => CompareOp::GreaterThan,
            Operator::GtEq => CompareOp::GreaterThanOrEqual,
            _ => unreachable!("`expression` only sends comparison operators here"),
        };
        Ok(JitProgram::Compare {
            op,
            left: Box::new(self.expression(left, &at(location, "left"))?),
            right: Box::new(self.expression(right, &at(location, "right"))?),
        })
    }

    /// Boolean `and`/`or`.
    pub(super) fn logical(
        &self,
        left: &Expr,
        op: Operator,
        right: &Expr,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let op = if matches!(op, Operator::And | Operator::LogicalAnd) {
            BooleanOp::And
        } else {
            BooleanOp::Or
        };
        Ok(JitProgram::Boolean {
            op,
            left: Box::new(self.boolean(left, &at(location, "left"))?),
            right: Box::new(self.boolean(right, &at(location, "right"))?),
        })
    }
}
