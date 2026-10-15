//! Translation of literals.

use polars_core::prelude::{AnyValue, DataType};
use polars_error::{PolarsResult, polars_bail};
use polars_plan::prelude::{DynLiteralValue, LiteralValue};

use super::Translator;
use crate::jit_program::{JitLiteral, JitProgram};

impl Translator<'_> {
    /// An Int64 literal.
    pub(super) fn literal(&self, value: &LiteralValue, location: &str) -> PolarsResult<JitProgram> {
        let value = match value {
            LiteralValue::Dyn(DynLiteralValue::Int(value)) => *value,
            LiteralValue::Scalar(value) => match value.as_any_value() {
                AnyValue::Int64(value) => value as i128,
                value => polars_bail!(InvalidOperation:
                    "literal {} is unsupported at {location}", value.dtype()),
            },
            _ => polars_bail!(InvalidOperation: "literal is unsupported at {location}"),
        };
        Ok(JitProgram::Literal {
            value: JitLiteral::Signed(value),
            dtype: DataType::Int64,
        })
    }
}
