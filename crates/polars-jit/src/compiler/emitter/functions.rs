//! Functions: `not`.

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::jit_program::JitProgram;

impl Emitter<'_> {
    /// `not`.
    pub(super) fn not(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        input: &JitProgram,
    ) -> PolarsResult<Value> {
        let value = self.emit(builder, input)?;
        Ok(builder.ins().bxor_imm_u(value, 1))
    }
}
