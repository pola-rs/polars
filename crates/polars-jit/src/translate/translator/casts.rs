//! Translation of explicit casts, with Polars' strict, non-strict, and wrapping overflow rules.

use polars_core::chunked_array::cast::CastOptions;
use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_plan::dsl::{DataTypeExpr, Expr};

use super::Translator;
use crate::jit_program::{CastOverflow, JitProgram, ScalarType};
use crate::translate::helpers::{at, is_null_literal};

impl Translator<'_> {
    pub(super) fn cast_expression(
        &self,
        input: &Expr,
        dtype: &DataTypeExpr,
        options: &CastOptions,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let DataTypeExpr::Literal(dtype) = dtype else {
            polars_bail!(InvalidOperation:
                "foldv only supports casts to concrete scalar dtypes at {location}")
        };
        let target = ScalarType::from_dtype(dtype)?;
        // An untyped null cast to a type is a null of that type. Any other literal keeps its own
        // type so the cast below can apply its overflow rule to the value.
        let expected = is_null_literal(input).then_some(target);
        let input = self.expression(input, expected, &at(location, "input"))?;
        polars_ensure!(input.dtype() == target
            || (input.dtype() != ScalarType::String && target != ScalarType::String),
            InvalidOperation: "foldv does not support casts from {} to {} at {location}",
            input.dtype().dtype(), target.dtype());
        // A cast to the input's own type changes nothing.
        if input.dtype() == target {
            return Ok(input);
        }
        // Only casts that can receive an out-of-range value need a runtime check.
        let overflow = if cast_stays_non_null(input.dtype(), target) {
            CastOverflow::Unchecked
        } else {
            match options {
                CastOptions::Strict => CastOverflow::Raise,
                CastOptions::NonStrict => CastOverflow::Null,
                CastOptions::Overflowing => CastOverflow::Unchecked,
            }
        };
        Ok(JitProgram::Cast {
            input: Box::new(input),
            dtype: target,
            overflow,
        })
    }
}

/// Whether every value of `source` has a value in `target`, so the cast can never overflow.
pub(super) fn cast_stays_non_null(source: ScalarType, target: ScalarType) -> bool {
    if source == target || target.is_float() || target == ScalarType::Boolean {
        return true;
    }
    if source.is_float() {
        return false;
    }
    let source_bits = source.bit_width();
    let target_bits = target.bit_width();
    (source.is_signed() == target.is_signed() && target_bits >= source_bits)
        || (!source.is_signed() && target.is_signed() && target_bits > source_bits)
}
