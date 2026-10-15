//! Translation of binary operators: arithmetic, comparisons, and Boolean `and`/`or`.

use polars_core::prelude::DataType;
use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::{Expr, Operator};

use super::Translator;
use crate::jit_program::{BinaryOp, BooleanOp, CompareOp, JitProgram, ScalarType};
use crate::translate::helpers::{at, common_type, with_arithmetic_types};

impl Translator<'_> {
    /// Numeric arithmetic; `+` also concatenates strings.
    pub(super) fn arithmetic(
        &self,
        expression: &Expr,
        left: &Expr,
        op: Operator,
        right: &Expr,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let mut operands = self.operands(left, right, location)?;
        let dtype = self.result_type(expression, operands.0.dtype(), operands.1.dtype())?;
        // Like Polars, `+` concatenates strings.
        let concatenates = dtype == ScalarType::String && matches!(op, Operator::Plus);
        polars_ensure!(dtype.is_numeric() || concatenates, InvalidOperation:
            "operator {op} requires numeric operands at {location}");
        if op == Operator::TrueDivide {
            // Polars coerces integer operands before converting them to floats. In particular,
            // a negative untyped literal makes a UInt64 operand Int64, nulling values above MAX.
            // Addition uses the same numeric coercion without true division's floating output.
            let common = with_arithmetic_types(&(left.clone() + right.clone()))
                .to_field(&self.schema)?
                .dtype;
            if matches!(
                common,
                DataType::Int32 | DataType::Int64 | DataType::UInt32 | DataType::UInt64
            ) {
                let (left, right) = self.cast_pair(operands, ScalarType::from_dtype(&common)?)?;
                operands = (*left, *right);
            }
        }
        let (left, right) = self.cast_pair(operands, dtype)?;
        let op = match op {
            Operator::Plus => BinaryOp::Add,
            Operator::Minus => BinaryOp::Subtract,
            Operator::Multiply => BinaryOp::Multiply,
            Operator::TrueDivide => BinaryOp::TrueDivide,
            // Polars' Rust `/` floors integers and true-divides floats.
            Operator::RustDivide if dtype.is_float() => BinaryOp::TrueDivide,
            Operator::RustDivide | Operator::FloorDivide => BinaryOp::FloorDivide,
            Operator::Modulus => BinaryOp::Modulus,
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
            Operator::EqValidity => CompareOp::EqualMissing,
            Operator::NotEqValidity => CompareOp::NotEqualMissing,
            _ => unreachable!("`expression` only sends comparison operators here"),
        };
        Ok(JitProgram::Compare { op, left, right })
    }

    /// Boolean `and`/`or`, with Polars' Kleene logic for nulls.
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
