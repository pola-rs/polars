//! Binary operators.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::jit_program::{BinaryOp, BooleanOp, CompareOp, JitProgram};

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

    /// Compare Int64 values.
    pub(super) fn compare_integers(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: CompareOp,
        left: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<Value> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        let condition = match op {
            CompareOp::Equal => IntCC::Equal,
            CompareOp::NotEqual => IntCC::NotEqual,
            CompareOp::LessThan => IntCC::SignedLessThan,
            CompareOp::LessThanOrEqual => IntCC::SignedLessThanOrEqual,
            CompareOp::GreaterThan => IntCC::SignedGreaterThan,
            CompareOp::GreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        };
        Ok(builder.ins().icmp(condition, left, right))
    }

    /// Boolean `and`/`or`.
    pub(super) fn logical(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: BooleanOp,
        left: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<Value> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        Ok(match op {
            BooleanOp::And => builder.ins().band(left, right),
            BooleanOp::Or => builder.ins().bor(left, right),
        })
    }
}
