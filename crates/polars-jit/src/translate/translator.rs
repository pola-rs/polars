//! The `Translator` turns a fold body into typed JIT nodes.
//!
//! `expression` is the single place that matches Polars expressions: every arm calls the
//! method that translates that expression. The methods live in the modules below, grouped by
//! kind.

mod literals;
mod operators;

use polars_core::prelude::{DataType, Schema};
use polars_error::{PolarsResult, polars_bail};
use polars_plan::dsl::{Expr, Operator};
use polars_utils::pl_str::PlSmallStr;

use super::helpers::unsupported;
use super::resolve::STATE_NAME;
use crate::jit_program::{InputId, JitProgram};

pub(super) struct Translator<'a> {
    /// The state and row-input dtypes, for asking Polars the dtype of a subexpression.
    schema: &'a Schema,
    row_name: &'a PlSmallStr,
}

impl<'a> Translator<'a> {
    pub(super) fn new(schema: &'a Schema, row_name: &'a PlSmallStr) -> Self {
        Self { schema, row_name }
    }

    /// Translate `expression`; `location` names it in error messages.
    pub(super) fn expression(&self, expression: &Expr, location: &str) -> PolarsResult<JitProgram> {
        match expression {
            Expr::Column(name) => self.column(name),
            Expr::Literal(value) => self.literal(value, location),
            Expr::Alias(inner, _) => self.expression(inner, location),

            // Operators, in `operators`.
            Expr::BinaryExpr {
                left,
                op: op @ (Operator::Plus | Operator::Minus | Operator::Multiply),
                right,
            } => self.arithmetic(expression, left, *op, right, location),
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
            dtype: DataType::Int64,
        })
    }
}
