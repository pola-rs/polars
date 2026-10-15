//! Translation of literals, which Polars types partly by context.

use polars_core::prelude::AnyValue;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_plan::prelude::{DynLiteralValue, LiteralValue};

use super::Translator;
use crate::jit_program::{JitLiteral, JitProgram, ScalarType};

impl Translator<'_> {
    pub(super) fn literal(
        &self,
        value: &LiteralValue,
        expected: Option<ScalarType>,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let (dtype, value) = match value {
            LiteralValue::Dyn(DynLiteralValue::Int(value)) => {
                if let Some(dtype) = expected.filter(|dtype| dtype.is_float()) {
                    let value = *value as f64;
                    let literal = if dtype == ScalarType::Float32 {
                        JitLiteral::Float32((value as f32).to_bits())
                    } else {
                        JitLiteral::Float64(value.to_bits())
                    };
                    return Ok(JitProgram::Literal {
                        value: literal,
                        dtype,
                    });
                }
                // Like Polars, an untyped integer adopts the other operand's integer type when the
                // value fits it, and is otherwise Int64, so `u32_column + (-1)` is Int64. Polars
                // uses Int128 beyond that, except that a standalone literal (such as the input of
                // a cast) may use UInt64, which converts the same way.
                let fits = |dtype: ScalarType| integer_fits(*value, dtype);
                let dtype = match expected {
                    Some(dtype) if dtype.is_numeric() && fits(dtype) => dtype,
                    _ if fits(ScalarType::Int64) => ScalarType::Int64,
                    None if fits(ScalarType::UInt64) => ScalarType::UInt64,
                    _ => polars_bail!(InvalidOperation:
                        "integer literal {value} at {location} needs Int128, which foldv does not support"),
                };
                if dtype.is_signed() {
                    (dtype, JitLiteral::Signed(*value))
                } else {
                    (dtype, JitLiteral::Unsigned(*value as u128))
                }
            },
            LiteralValue::Dyn(DynLiteralValue::Float(value)) => {
                // Like Polars, an untyped float adopts a float operand's type but is Float64 next
                // to an integer, so `int_column + 0.5` becomes Float64.
                let dtype = expected
                    .filter(|dtype| dtype.is_float())
                    .unwrap_or(ScalarType::Float64);
                let literal = if dtype == ScalarType::Float32 {
                    JitLiteral::Float32((*value as f32).to_bits())
                } else {
                    JitLiteral::Float64(value.to_bits())
                };
                (dtype, literal)
            },
            LiteralValue::Dyn(DynLiteralValue::Str(value)) => {
                (ScalarType::String, JitLiteral::String(value.clone()))
            },
            LiteralValue::Scalar(value) => {
                if matches!(value.as_any_value(), AnyValue::Null) {
                    // A typed null such as `pl.lit(None, dtype=pl.Int64)` keeps its dtype; only an
                    // untyped null takes the type of what it is combined with.
                    let dtype = if value.dtype().is_null() {
                        expected.ok_or_else(|| {
                            polars_err!(InvalidOperation:
                                "null literal at {location} needs a concrete dtype")
                        })?
                    } else {
                        ScalarType::from_dtype(value.dtype())?
                    };
                    return Ok(JitProgram::Null { dtype });
                }
                let dtype = ScalarType::from_dtype(value.dtype())?;
                let literal = match value.as_any_value() {
                    AnyValue::Boolean(v) => JitLiteral::Boolean(v),
                    AnyValue::Int32(v) => JitLiteral::Signed(v as i128),
                    AnyValue::Int64(v) => JitLiteral::Signed(v as i128),
                    AnyValue::UInt32(v) => JitLiteral::Unsigned(v as u128),
                    AnyValue::UInt64(v) => JitLiteral::Unsigned(v as u128),
                    AnyValue::Float32(v) => JitLiteral::Float32(v.to_bits()),
                    AnyValue::Float64(v) => JitLiteral::Float64(v.to_bits()),
                    AnyValue::String(v) => JitLiteral::String(v.into()),
                    AnyValue::StringOwned(v) => JitLiteral::String(v),
                    value => polars_bail!(InvalidOperation:
                        "literal {} is unsupported at {location}", value.dtype()),
                };
                (dtype, literal)
            },
            _ => polars_bail!(InvalidOperation: "literal is unsupported at {location}"),
        };
        Ok(JitProgram::Literal { value, dtype })
    }
}

/// Whether an integer literal is representable in an integer `dtype`.
fn integer_fits(value: i128, dtype: ScalarType) -> bool {
    let bits = u32::from(dtype.bit_width());
    if dtype.is_signed() {
        let max = (1_i128 << (bits - 1)) - 1;
        (-max - 1..=max).contains(&value)
    } else {
        (0..=(1_i128 << bits) - 1).contains(&value)
    }
}
