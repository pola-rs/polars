//! Literals.

use cranelift_codegen::ir::{InstBuilder, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::{PolarsResult, polars_err};

use super::Emitter;
use crate::jit_program::JitLiteral;

impl Emitter<'_> {
    /// An Int64 literal.
    pub(super) fn literal(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        literal: &JitLiteral,
    ) -> PolarsResult<Value> {
        let JitLiteral::Signed(value) = literal;
        Ok(builder.ins().iconst(
            types::I64,
            i64::try_from(*value).map_err(
                |_| polars_err!(InvalidOperation: "integer literal does not fit in Int64"),
            )?,
        ))
    }
}
