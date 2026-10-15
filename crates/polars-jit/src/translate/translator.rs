//! The `Translator` turns a fold body into typed JIT nodes.
//!
//! `expression` is the single place that matches Polars expressions: every arm calls the
//! method that translates that expression or function. The methods live in the modules below,
//! grouped by kind, and `typing` holds the rules that give operands a common type.

mod casts;
mod conditionals;
mod functions;
mod literals;
mod operators;
mod typing;

use polars_error::{PolarsResult, polars_bail};
use polars_plan::dsl::{BooleanFunction, Expr, FunctionExpr, Operator};
use polars_utils::pl_str::PlSmallStr;

use super::helpers::unsupported;
use super::resolve::{ResolvedTypes, STATE_NAME};
use crate::jit_program::{InputId, JitProgram, ScalarType};

pub(super) struct Translator<'a> {
    row_name: &'a PlSmallStr,
    state_type: ScalarType,
    row_type: ScalarType,
}

impl<'a> Translator<'a> {
    pub(super) fn new(row_name: &'a PlSmallStr, resolved: &ResolvedTypes) -> Self {
        Self {
            row_name,
            state_type: resolved.state_type,
            row_type: resolved.row_type,
        }
    }

    /// Translate `expression`. An untyped literal in it takes the `expected` type when that
    /// type can hold it; `location` names the expression in error messages.
    pub(super) fn expression(
        &self,
        expression: &Expr,
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        match expression {
            Expr::Column(name) => self.column(name),
            Expr::Literal(value) => self.literal(value, expected, location),
            Expr::Alias(inner, _) => self.expression(inner, expected, location),
            Expr::Cast { expr, dtype, .. } => self.cast_expression(expr, dtype, location),

            // Operators, in `operators`.
            Expr::BinaryExpr {
                left,
                op: op @ (Operator::Plus | Operator::Minus | Operator::Multiply),
                right,
            } => self.arithmetic(left, *op, right, location),
            Expr::BinaryExpr { left, op, right } if op.is_comparison() => {
                self.comparison(left, *op, right, location)
            },
            Expr::BinaryExpr {
                left,
                op: op @ (Operator::And | Operator::LogicalAnd | Operator::Or | Operator::LogicalOr),
                right,
            } => self.logical(left, *op, right, location),

            // Functions, in `functions`.
            Expr::Function {
                input,
                function: FunctionExpr::Boolean(BooleanFunction::Not),
            } => self.not(expression, input, location),

            // Ternary, `when(...).then(...).otherwise(...)`, in `conditionals`.
            Expr::Ternary {
                predicate,
                truthy,
                falsy,
            } => self.ternary(predicate, truthy, falsy, expected, location),
            _ => Err(unsupported(expression, location)),
        }
    }

    /// The carried state or the row input.
    fn column(&self, name: &PlSmallStr) -> PolarsResult<JitProgram> {
        Ok(JitProgram::Input {
            input: if name.as_str() == STATE_NAME {
                InputId(0)
            } else if name == self.row_name {
                InputId(1)
            } else {
                polars_bail!(ColumnNotFound: "JIT input `{name}`")
            },
            dtype: if name.as_str() == STATE_NAME {
                self.state_type
            } else {
                self.row_type
            },
        })
    }
}
