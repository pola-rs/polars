//! Numeric functions, with operand and result types resolved using Polars' rules.

use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::Expr;

use super::Translator;
use super::casts::cast_stays_non_null;
use crate::jit_program::{CastOverflow, JitProgram, NumericUnary, ScalarType};
use crate::translate::helpers::{at, inputs};

impl Translator<'_> {
    pub(super) fn numeric_unary(
        &self,
        expression: &Expr,
        input: &[Expr],
        op: NumericUnary,
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let [value] = inputs(input, expression, location)?;
        let value = self.expression(value, expected, &at(location, "input"))?;
        polars_ensure!(value.dtype().is_numeric(), InvalidOperation:
            "numeric operation at {location} requires numeric input, got {}", value.dtype().dtype());
        polars_ensure!(op != NumericUnary::Negate || value.dtype().is_signed() || value.dtype().is_float(),
            InvalidOperation: "negation is not supported for dtype {} at {location}", value.dtype().dtype());
        let dtype = self.result_type(expression, value.dtype(), value.dtype())?;
        Ok(JitProgram::NumericUnary {
            op,
            input: Box::new(self.cast(value, dtype)?),
            dtype,
        })
    }

    pub(super) fn pow(
        &self,
        expression: &Expr,
        input: &[Expr],
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let [base, exponent] = inputs(input, expression, location)?;
        let base = self.expression(base, expected, &at(location, "base"))?;
        let exponent = self.expression(exponent, None, &at(location, "exponent"))?;
        polars_ensure!(base.dtype().is_numeric() && exponent.dtype().is_numeric(), InvalidOperation:
            "pow requires numeric operands at {location}");
        let dtype = self.result_type(expression, base.dtype(), exponent.dtype())?;
        // Native integer powers require a nonnegative exponent that fits UInt32.
        let exponent_type = if dtype.is_float() {
            dtype
        } else {
            ScalarType::UInt32
        };
        Ok(JitProgram::Pow {
            base: Box::new(self.cast(base, dtype)?),
            exponent: Box::new(strict_cast(exponent, exponent_type)),
            dtype,
        })
    }

    pub(super) fn clip(
        &self,
        _expression: &Expr,
        input: &[Expr],
        bounds: (bool, bool),
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let (has_min, has_max) = bounds;
        polars_ensure!(input.len() == 1 + usize::from(has_min) + usize::from(has_max), InvalidOperation:
            "invalid clip inputs at {location}");
        let value = self.expression(&input[0], expected, &at(location, "input"))?;
        let dtype = value.dtype();
        polars_ensure!(dtype.is_numeric(), InvalidOperation: "clip requires numeric input at {location}");
        let bound = |index, name| {
            let value = self.expression(&input[index], Some(dtype), &at(location, name))?;
            polars_ensure!(value.dtype() != ScalarType::String, InvalidOperation:
                "foldv does not support casts from str to {} at {}", dtype.dtype(), at(location, name));
            Ok(Box::new(strict_cast(value, dtype)))
        };
        Ok(JitProgram::Clip {
            input: Box::new(value),
            min: has_min.then(|| bound(1, "min")).transpose()?,
            max: has_max
                .then(|| bound(1 + usize::from(has_min), "max"))
                .transpose()?,
            dtype,
        })
    }

    pub(super) fn min_max_horizontal(
        &self,
        expression: &Expr,
        input: &[Expr],
        max: bool,
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        polars_ensure!(!input.is_empty(), InvalidOperation: "horizontal min/max requires inputs at {location}");
        let field = expression.to_field(&self.schema)?;
        let dtype = if field.dtype.is_null() {
            expected.unwrap_or(self.state_type)
        } else {
            ScalarType::from_dtype(&field.dtype.materialize_unknown(false)?)?
        };
        polars_ensure!(dtype.is_numeric(), InvalidOperation: "horizontal min/max requires numeric inputs at {location}");
        let mut values = input.iter().enumerate().map(|(index, value)| {
            let value = self.expression(
                value,
                Some(dtype),
                &at(location, &format!("input[{index}]")),
            )?;
            self.cast(value, dtype)
        });
        let mut result = values.next().unwrap()?;
        for value in values {
            result = JitProgram::MinMax {
                left: Box::new(result),
                right: Box::new(value?),
                max,
                dtype,
            };
        }
        Ok(result)
    }
}

fn strict_cast(input: JitProgram, dtype: ScalarType) -> JitProgram {
    if input.dtype() == dtype {
        return input;
    }
    let overflow = if cast_stays_non_null(input.dtype(), dtype) {
        CastOverflow::Unchecked
    } else {
        CastOverflow::Raise
    };
    JitProgram::Cast {
        input: Box::new(input),
        dtype,
        overflow,
    }
}
