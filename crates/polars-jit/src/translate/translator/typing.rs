//! Typing rules shared by the operations that combine two values.

use polars_error::{PolarsResult, polars_ensure};
use polars_plan::dsl::Expr;

use super::Translator;
use crate::jit_program::{CastOverflow, JitProgram, ScalarType};
use crate::translate::helpers::common_type;

impl Translator<'_> {
    /// Translate two expressions whose types depend on each other: the one translated second
    /// takes the first one's type as its expected type. `second_first` translates `second`
    /// first, for when `first` is an untyped literal that needs a type from `second`.
    pub(super) fn translate_pair(
        &self,
        (first, first_location): (&Expr, &str),
        (second, second_location): (&Expr, &str),
        expected: Option<ScalarType>,
        second_first: bool,
    ) -> PolarsResult<(JitProgram, JitProgram)> {
        Ok(if second_first {
            let second = self.expression(second, expected, second_location)?;
            let first = self.expression(first, Some(second.dtype()), first_location)?;
            (first, second)
        } else {
            let first = self.expression(first, expected, first_location)?;
            let second = self.expression(second, Some(first.dtype()), second_location)?;
            (first, second)
        })
    }

    /// The dtype Polars gives `expression`. Polars types untyped literals by context, so
    /// `u64_column + (-1)` is Int64 even though the supertype of UInt64 and Int64 is Float64.
    /// An expression Polars types as `Null` falls back to the operands' common type.
    pub(super) fn result_type(
        &self,
        expression: &Expr,
        left: ScalarType,
        right: ScalarType,
    ) -> PolarsResult<ScalarType> {
        let dtype = expression.to_field(&self.schema)?.dtype;
        if dtype.is_null() {
            common_type(left, right)
        } else {
            ScalarType::from_dtype(&dtype)
        }
    }

    /// Convert `input` to `dtype` for an operation that combines it with another value. Such
    /// conversions never fail; explicit casts are translated in `casts`.
    pub(super) fn cast(&self, input: JitProgram, dtype: ScalarType) -> PolarsResult<JitProgram> {
        if input.dtype() == dtype {
            Ok(input)
        } else {
            polars_ensure!(input.dtype().is_numeric() && dtype.is_numeric(), InvalidOperation:
                "foldv cannot cast {} to {}", input.dtype().dtype(), dtype.dtype());
            Ok(JitProgram::Cast {
                input: Box::new(input),
                dtype,
                overflow: CastOverflow::Unchecked,
            })
        }
    }

    /// Convert both values of a pair to `dtype`.
    pub(super) fn cast_pair(
        &self,
        (first, second): (JitProgram, JitProgram),
        dtype: ScalarType,
    ) -> PolarsResult<(Box<JitProgram>, Box<JitProgram>)> {
        Ok((
            Box::new(self.cast(first, dtype)?),
            Box::new(self.cast(second, dtype)?),
        ))
    }

    /// Translate an expression that must produce a Boolean, such as a predicate.
    pub(super) fn boolean(&self, expression: &Expr, location: &str) -> PolarsResult<JitProgram> {
        let program = self.expression(expression, Some(ScalarType::Boolean), location)?;
        polars_ensure!(program.dtype() == ScalarType::Boolean, InvalidOperation:
            "predicate at {location} must produce Boolean");
        Ok(program)
    }

    /// Translate an expression that must produce a string.
    pub(super) fn string(&self, expression: &Expr, location: &str) -> PolarsResult<JitProgram> {
        let program = self.expression(expression, Some(ScalarType::String), location)?;
        polars_ensure!(program.dtype() == ScalarType::String, InvalidOperation:
            "foldv string operation at {location} requires str, got {}", program.dtype().dtype());
        Ok(program)
    }
}
