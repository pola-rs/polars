//! The `Translator` turns a fold body into typed JIT nodes.
//!
//! `expression` is the single place that matches Polars expressions: every arm calls the
//! method that translates that expression or function. The methods live in the modules below,
//! grouped by kind, and `typing` holds the rules that give operands a common type.

mod casts;
mod conditionals;
mod functions;
mod literals;
mod numbers;
mod operators;
mod typing;

use polars_core::prelude::Schema;
use polars_error::{PolarsResult, polars_err};
use polars_plan::dsl::{
    BooleanFunction, Expr, FunctionExpr, Operator, PowFunction, StringFunction,
};
use polars_utils::pl_str::PlSmallStr;

use super::helpers::unsupported;
use super::resolve::{ResolvedTypes, STATE_NAME, body_schema};
use crate::jit_program::{InputId, JitProgram, NumericUnary, ScalarType};

pub(super) struct Translator<'a> {
    row_names: &'a [PlSmallStr],
    state_type: ScalarType,
    row_types: &'a [ScalarType],
    /// The state and row-input dtypes, for asking Polars the dtype of a subexpression.
    schema: Schema,
}

impl<'a> Translator<'a> {
    pub(super) fn new(row_names: &'a [PlSmallStr], resolved: &'a ResolvedTypes) -> Self {
        let row_dtypes = resolved
            .row_types
            .iter()
            .map(|dtype| dtype.dtype())
            .collect::<Vec<_>>();
        Self {
            row_names,
            state_type: resolved.state_type,
            row_types: &resolved.row_types,
            schema: body_schema(row_names, &resolved.state_type.dtype(), &row_dtypes),
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
            Expr::Cast {
                expr,
                dtype,
                options,
            } => self.cast_expression(expr, dtype, options, location),

            // Operators, in `operators`.
            Expr::BinaryExpr {
                left,
                op:
                    op @ (Operator::Plus
                    | Operator::Minus
                    | Operator::Multiply
                    | Operator::TrueDivide
                    | Operator::RustDivide
                    | Operator::FloorDivide
                    | Operator::Modulus),
                right,
            } => self.arithmetic(expression, left, *op, right, location),
            Expr::BinaryExpr { left, op, right } if op.is_comparison() => {
                self.comparison(left, *op, right, location)
            },
            Expr::BinaryExpr {
                left,
                op: op @ (Operator::And | Operator::LogicalAnd | Operator::Or | Operator::LogicalOr),
                right,
            } => self.logical(left, *op, right, location),

            // Functions, in `functions`.
            Expr::Function { input, function } => match function {
                FunctionExpr::Negate => {
                    self.numeric_unary(expression, input, NumericUnary::Negate, expected, location)
                },
                FunctionExpr::Abs => {
                    self.numeric_unary(expression, input, NumericUnary::Abs, expected, location)
                },
                FunctionExpr::Sign => {
                    self.numeric_unary(expression, input, NumericUnary::Sign, expected, location)
                },
                FunctionExpr::Floor => {
                    self.numeric_unary(expression, input, NumericUnary::Floor, expected, location)
                },
                FunctionExpr::Ceil => {
                    self.numeric_unary(expression, input, NumericUnary::Ceil, expected, location)
                },
                FunctionExpr::Round { decimals, mode } => self.numeric_unary(
                    expression,
                    input,
                    NumericUnary::Round {
                        decimals: *decimals,
                        mode: *mode,
                    },
                    expected,
                    location,
                ),
                FunctionExpr::Pow(PowFunction::Sqrt) => {
                    self.numeric_unary(expression, input, NumericUnary::Sqrt, expected, location)
                },
                FunctionExpr::Pow(PowFunction::Generic) => {
                    self.pow(expression, input, expected, location)
                },
                FunctionExpr::Clip { has_min, has_max } => {
                    self.clip(expression, input, (*has_min, *has_max), expected, location)
                },
                FunctionExpr::MinHorizontal | FunctionExpr::MaxHorizontal => self
                    .min_max_horizontal(
                        expression,
                        input,
                        matches!(function, FunctionExpr::MaxHorizontal),
                        expected,
                        location,
                    ),
                FunctionExpr::Boolean(BooleanFunction::Not) => {
                    self.not(expression, input, location)
                },
                FunctionExpr::Boolean(BooleanFunction::IsNull) => {
                    self.is_null(expression, input, false, location)
                },
                FunctionExpr::Boolean(BooleanFunction::IsNotNull) => {
                    self.is_null(expression, input, true, location)
                },
                FunctionExpr::Boolean(
                    function @ (BooleanFunction::IsNan
                    | BooleanFunction::IsNotNan
                    | BooleanFunction::IsFinite
                    | BooleanFunction::IsInfinite),
                ) => self.float_check(expression, function, input, location),
                FunctionExpr::FillNull => self.fill_null(expression, input, expected, location),
                FunctionExpr::StringExpr(StringFunction::LenBytes) => {
                    self.string_length(expression, input, false, location)
                },
                FunctionExpr::StringExpr(StringFunction::LenChars) => {
                    self.string_length(expression, input, true, location)
                },
                FunctionExpr::StringExpr(StringFunction::ConcatHorizontal {
                    delimiter,
                    ignore_nulls,
                }) => self.concat_str(expression, input, delimiter, *ignore_nulls, location),
                _ => Err(unsupported(expression, location)),
            },

            // Ternary, `when(...).then(...).otherwise(...)`, in `conditionals`.
            Expr::Ternary {
                predicate,
                truthy,
                falsy,
            } => self.ternary(expression, predicate, truthy, falsy, expected, location),
            _ => Err(unsupported(expression, location)),
        }
    }

    /// The carried state or a row input.
    fn column(&self, name: &PlSmallStr) -> PolarsResult<JitProgram> {
        if name.as_str() == STATE_NAME {
            return Ok(JitProgram::Input {
                input: InputId(0),
                dtype: self.state_type,
            });
        }
        let index = self
            .row_names
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| polars_err!(ColumnNotFound: "JIT input `{name}`"))?;
        Ok(JitProgram::Input {
            input: InputId(index as u32 + 1),
            dtype: self.row_types[index],
        })
    }
}
