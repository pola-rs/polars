//! Translation of Polars function expressions such as `is_null()` or `str.len_chars()`.
//!
//! Polars represents almost every method call as one `Expr::Function` node whose
//! `FunctionExpr` names the function; `Translator::expression` matches the function and calls
//! its implementation here.

use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::{BooleanFunction, Expr};
use polars_utils::pl_str::PlSmallStr;

use super::Translator;
use crate::jit_program::{BinaryOp, FloatCheck, JitLiteral, JitProgram, ScalarType};
use crate::translate::helpers::{at, inputs, is_null_literal, negated_if, unsupported};

impl Translator<'_> {
    /// `not`.
    pub(super) fn not(
        &self,
        expression: &Expr,
        input: &[Expr],
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let [value] = inputs(input, expression, location)?;
        Ok(JitProgram::Not(Box::new(
            self.boolean(value, &at(location, "input"))?,
        )))
    }

    /// `is_null`, or `is_not_null` when `negate` is set.
    pub(super) fn is_null(
        &self,
        expression: &Expr,
        input: &[Expr],
        negate: bool,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let [value] = inputs(input, expression, location)?;
        // An untyped null literal is null whatever its dtype, so any dtype will do.
        let expected = is_null_literal(value).then_some(ScalarType::Boolean);
        let value = self.expression(value, expected, &at(location, "input"))?;
        Ok(negated_if(negate, JitProgram::IsNull(Box::new(value))))
    }

    /// `is_nan`, `is_not_nan`, `is_finite`, and `is_infinite`. `fill_nan` reaches here too,
    /// because Polars expands it to a `when` on `is_not_nan`.
    pub(super) fn float_check(
        &self,
        expression: &Expr,
        function: &BooleanFunction,
        input: &[Expr],
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let [value] = inputs(input, expression, location)?;
        let value = self.expression(value, None, &at(location, "input"))?;
        polars_ensure!(value.dtype().is_numeric(), InvalidOperation:
            "`{function}` operation not supported for dtype `{}`", value.dtype().dtype());
        let check = match function {
            BooleanFunction::IsNan | BooleanFunction::IsNotNan => FloatCheck::Nan,
            BooleanFunction::IsFinite => FloatCheck::Finite,
            _ => FloatCheck::Infinite,
        };
        Ok(negated_if(
            matches!(function, BooleanFunction::IsNotNan),
            JitProgram::FloatCheck {
                check,
                input: Box::new(value),
            },
        ))
    }

    /// `fill_null`, which is `when(value.is_null()).then(fill).otherwise(value)` with the type
    /// Polars gives the expression.
    pub(super) fn fill_null(
        &self,
        expression: &Expr,
        input: &[Expr],
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let [value, fill] = inputs(input, expression, location)?;
        // A null literal value takes its dtype from the fill, so translate the fill first.
        let operands = self.translate_pair(
            (value, &at(location, "input")),
            (fill, &at(location, "fill")),
            expected,
            is_null_literal(value),
        )?;
        let dtype = self.result_type(expression, operands.0.dtype(), operands.1.dtype())?;
        let (value, fill) = self.cast_pair(operands, dtype)?;
        Ok(JitProgram::FillNull { value, fill, dtype })
    }

    /// `str.len_bytes()`, or `str.len_chars()` when `chars` is set.
    pub(super) fn string_length(
        &self,
        expression: &Expr,
        input: &[Expr],
        chars: bool,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let [value] = inputs(input, expression, location)?;
        let value = self.string(value, &at(location, "input"))?;
        Ok(JitProgram::StringLength {
            input: Box::new(value),
            chars,
        })
    }

    /// `concat_str`, which joins its inputs with `delimiter`.
    pub(super) fn concat_str(
        &self,
        expression: &Expr,
        input: &[Expr],
        delimiter: &PlSmallStr,
        ignore_nulls: bool,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        if input.is_empty() {
            return Err(unsupported(expression, location));
        }
        // Without `ignore_nulls`, any null input makes the result null, like `+`.
        polars_ensure!(!ignore_nulls, InvalidOperation:
            "foldv does not support concat_str with ignore_nulls=True at {location}");
        let mut parts = input
            .iter()
            .enumerate()
            .map(|(index, part)| self.string(part, &at(location, &format!("input[{index}]"))));
        let mut joined = parts.next().expect("concat_str has at least one input")?;
        for part in parts {
            if !delimiter.is_empty() {
                joined = concatenate(
                    joined,
                    JitProgram::Literal {
                        value: JitLiteral::String(delimiter.clone()),
                        dtype: ScalarType::String,
                    },
                );
            }
            joined = concatenate(joined, part?);
        }
        Ok(joined)
    }
}

/// `left + right` on strings, which is null when either side is null.
fn concatenate(left: JitProgram, right: JitProgram) -> JitProgram {
    JitProgram::Binary {
        op: BinaryOp::Add,
        left: Box::new(left),
        right: Box::new(right),
        dtype: ScalarType::String,
    }
}
