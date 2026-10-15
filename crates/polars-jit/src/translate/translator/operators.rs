//! Translation of binary operators: arithmetic, comparisons, and Boolean `and`/`or`.

use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::{Expr, Operator};

use super::Translator;
use crate::jit_program::{BinaryOp, BooleanOp, CompareOp, JitProgram};
use crate::translate::helpers::{at, common_type};

impl Translator<'_> {
    /// `+`, `-`, and `*`, which convert both operands to their common type.
    pub(super) fn arithmetic(
        &self,
        left: &Expr,
        op: Operator,
        right: &Expr,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let operands = self.operands(left, right, location)?;
        let dtype = common_type(operands.0.dtype(), operands.1.dtype())?;
        let (left, right) = self.cast_pair(operands, dtype)?;
        polars_ensure!(dtype.is_numeric(), InvalidOperation:
            "operator {op} requires numeric operands at {location}");
        let op = match op {
            Operator::Plus => BinaryOp::Add,
            Operator::Minus => BinaryOp::Subtract,
            Operator::Multiply => BinaryOp::Multiply,
            _ => unreachable!("`expression` only sends arithmetic operators here"),
        };
        Ok(JitProgram::Binary {
            op,
            left,
            right,
            dtype,
        })
    }

    /// Comparisons, which convert both operands to their common type.
    pub(super) fn comparison(
        &self,
        left: &Expr,
        op: Operator,
        right: &Expr,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let operands = self.operands(left, right, location)?;
        let dtype = common_type(operands.0.dtype(), operands.1.dtype())?;
        let (left, right) = self.cast_pair(operands, dtype)?;
        let op = match op {
            Operator::Eq => CompareOp::Equal,
            Operator::NotEq => CompareOp::NotEqual,
            Operator::Lt => CompareOp::LessThan,
            Operator::LtEq => CompareOp::LessThanOrEqual,
            Operator::Gt => CompareOp::GreaterThan,
            Operator::GtEq => CompareOp::GreaterThanOrEqual,
            _ => unreachable!("`expression` only sends comparison operators here"),
        };
        Ok(JitProgram::Compare { op, left, right })
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

    /// Translate the operands of a binary operator. An untyped literal takes its type from the
    /// other operand, so a literal on the left is translated after the right.
    fn operands(
        &self,
        left: &Expr,
        right: &Expr,
        location: &str,
    ) -> PolarsResult<(JitProgram, JitProgram)> {
        let literal_first = matches!(left, Expr::Literal(_)) && !matches!(right, Expr::Literal(_));
        self.translate_pair(
            (left, &at(location, "left")),
            (right, &at(location, "right")),
            None,
            literal_first,
        )
    }
}
