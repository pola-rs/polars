//! Binary operators: arithmetic, comparisons, and Boolean `and`/`or`.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::jit_program::{BinaryOp, BooleanOp, CompareOp, JitProgram, ScalarType};

impl Emitter<'_> {
    /// `+`, `-`, and `*`.
    pub(super) fn arithmetic(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: BinaryOp,
        left: &JitProgram,
        right: &JitProgram,
        dtype: ScalarType,
    ) -> PolarsResult<Value> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        Ok(if dtype.is_float() {
            match op {
                BinaryOp::Add => builder.ins().fadd(left, right),
                BinaryOp::Subtract => builder.ins().fsub(left, right),
                BinaryOp::Multiply => builder.ins().fmul(left, right),
            }
        } else {
            match op {
                BinaryOp::Add => builder.ins().iadd(left, right),
                BinaryOp::Subtract => builder.ins().isub(left, right),
                BinaryOp::Multiply => builder.ins().imul(left, right),
            }
        })
    }

    /// Compare floats.
    pub(super) fn compare_floats(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: CompareOp,
        left: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<Value> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        let condition = match op {
            CompareOp::Equal => FloatCC::Equal,
            CompareOp::NotEqual => FloatCC::NotEqual,
            CompareOp::LessThan => FloatCC::LessThan,
            CompareOp::LessThanOrEqual => FloatCC::LessThanOrEqual,
            CompareOp::GreaterThan => FloatCC::GreaterThan,
            CompareOp::GreaterThanOrEqual => FloatCC::GreaterThanOrEqual,
        };
        Ok(builder.ins().fcmp(condition, left, right))
    }

    /// Compare integers or Booleans, signed or unsigned according to the operands' type.
    pub(super) fn compare_integers(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: CompareOp,
        left_program: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<Value> {
        let left = self.emit(builder, left_program)?;
        let right = self.emit(builder, right)?;
        let condition = match (op, left_program.dtype().is_signed()) {
            (CompareOp::Equal, _) => IntCC::Equal,
            (CompareOp::NotEqual, _) => IntCC::NotEqual,
            (CompareOp::LessThan, true) => IntCC::SignedLessThan,
            (CompareOp::LessThan, false) => IntCC::UnsignedLessThan,
            (CompareOp::LessThanOrEqual, true) => IntCC::SignedLessThanOrEqual,
            (CompareOp::LessThanOrEqual, false) => IntCC::UnsignedLessThanOrEqual,
            (CompareOp::GreaterThan, true) => IntCC::SignedGreaterThan,
            (CompareOp::GreaterThan, false) => IntCC::UnsignedGreaterThan,
            (CompareOp::GreaterThanOrEqual, true) => IntCC::SignedGreaterThanOrEqual,
            (CompareOp::GreaterThanOrEqual, false) => IntCC::UnsignedGreaterThanOrEqual,
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
