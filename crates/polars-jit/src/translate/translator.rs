//! The `Translator` turns a fold body into typed JIT nodes.
//!
//! `expression` is the single place that matches Polars expressions: every arm calls the
//! method that translates that expression. The methods live in the modules below, grouped by
//! kind.

mod operators;

use polars_error::PolarsResult;
use polars_plan::dsl::{Expr, Operator};
use polars_utils::pl_str::PlSmallStr;

use super::helpers::unsupported;
use crate::jit_program::JitProgram;

pub(super) struct Translator<'a> {
    row_names: &'a [PlSmallStr],
}

impl<'a> Translator<'a> {
    pub(super) fn new(row_names: &'a [PlSmallStr]) -> Self {
        Self { row_names }
    }

    /// Translate `expression`.
    pub(super) fn expression(&self, expression: &Expr) -> PolarsResult<JitProgram> {
        match expression {
            // Operators, in `operators`.
            Expr::BinaryExpr {
                left,
                op: Operator::Plus,
                right,
            } => self.arithmetic(left, right),
            _ => Err(unsupported()),
        }
    }
}
