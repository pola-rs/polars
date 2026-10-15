//! Literals.

use cranelift_codegen::ir::{InstBuilder, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::{PolarsResult, polars_err};

use super::Emitter;
use crate::jit_program::JitLiteral;

impl Emitter<'_> {
    /// A Boolean or Int64 literal.
    pub(super) fn literal(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        literal: &JitLiteral,
    ) -> PolarsResult<Value> {
        Ok(match literal {
            JitLiteral::Boolean(value) => builder.ins().iconst(types::I8, i64::from(*value)),
            JitLiteral::Signed(value) => builder.ins().iconst(
                types::I64,
                i64::try_from(*value).map_err(
                    |_| polars_err!(InvalidOperation: "integer literal does not fit in Int64"),
                )?,
            ),
        })
    }
}
