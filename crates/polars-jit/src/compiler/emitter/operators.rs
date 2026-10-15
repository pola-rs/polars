//! Binary operators.

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::jit_program::{BinaryOp, JitProgram};

impl Emitter<'_> {
    /// `+`, `-`, and `*` on Int64 values.
    pub(super) fn arithmetic(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: BinaryOp,
        left: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<Value> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        Ok(match op {
            BinaryOp::Add => builder.ins().iadd(left, right),
            BinaryOp::Subtract => builder.ins().isub(left, right),
            BinaryOp::Multiply => builder.ins().imul(left, right),
        })
    }
}
