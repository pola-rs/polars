//! Translation of literals, which Polars types partly by context.

use polars_core::prelude::AnyValue;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
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
                let dtype = expected.unwrap_or(ScalarType::Int64);
                if dtype.is_float() {
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
                polars_ensure!(dtype.is_numeric(), InvalidOperation:
                    "integer literal at {location} cannot be used as {}", dtype.dtype());
                if dtype.is_signed() {
                    (dtype, JitLiteral::Signed(*value))
                } else {
                    (dtype, JitLiteral::Unsigned(u128::try_from(*value).map_err(|_| {
                        polars_err!(InvalidOperation:
                            "negative literal at {location} cannot be represented as {}", dtype.dtype())
                    })?))
                }
            },
            LiteralValue::Dyn(DynLiteralValue::Float(value)) => {
                let dtype = expected.unwrap_or(ScalarType::Float64);
                let literal = match dtype {
                    ScalarType::Float32 if !value.is_nan() => {
                        JitLiteral::Float32((*value as f32).to_bits())
                    },
                    ScalarType::Float64 if !value.is_nan() => JitLiteral::Float64(value.to_bits()),
                    ScalarType::Float32 | ScalarType::Float64 => polars_bail!(InvalidOperation:
                        "foldv does not support NaN values yet"),
                    _ => polars_bail!(InvalidOperation:
                        "floating literal at {location} cannot be used as {}", dtype.dtype()),
                };
                (dtype, literal)
            },
            LiteralValue::Scalar(value) => {
                let dtype = ScalarType::from_dtype(value.dtype())?;
                let literal = match value.as_any_value() {
                    AnyValue::Boolean(v) => JitLiteral::Boolean(v),
                    AnyValue::Int32(v) => JitLiteral::Signed(v as i128),
                    AnyValue::Int64(v) => JitLiteral::Signed(v as i128),
                    AnyValue::UInt32(v) => JitLiteral::Unsigned(v as u128),
                    AnyValue::UInt64(v) => JitLiteral::Unsigned(v as u128),
                    AnyValue::Float32(v) if !v.is_nan() => JitLiteral::Float32(v.to_bits()),
                    AnyValue::Float64(v) if !v.is_nan() => JitLiteral::Float64(v.to_bits()),
                    AnyValue::Float32(_) | AnyValue::Float64(_) => polars_bail!(InvalidOperation:
                        "foldv does not support NaN values yet"),
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
