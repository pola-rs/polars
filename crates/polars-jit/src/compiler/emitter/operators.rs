//! Binary operators: arithmetic, comparisons, and Boolean `and`/`or`.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::values::JitValue;
use crate::jit_program::{BinaryOp, BooleanOp, CompareOp, JitProgram, ScalarType};

impl Emitter<'_> {
    /// `+`, `-`, and `*`; null when either operand is null.
    pub(super) fn arithmetic(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: BinaryOp,
        left: &JitProgram,
        right: &JitProgram,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        let (l, r) = (left.data, right.data);
        let data = if dtype.is_float() {
            match op {
                BinaryOp::Add => builder.ins().fadd(l, r),
                BinaryOp::Subtract => builder.ins().fsub(l, r),
                BinaryOp::Multiply => builder.ins().fmul(l, r),
            }
        } else {
            match op {
                BinaryOp::Add => builder.ins().iadd(l, r),
                BinaryOp::Subtract => builder.ins().isub(l, r),
                BinaryOp::Multiply => builder.ins().imul(l, r),
            }
        };
        Ok(JitValue {
            data,
            valid: builder.ins().band(left.valid, right.valid),
        })
    }

    /// Compare floats in Polars' total order, where NaN equals NaN and sorts above every other
    /// value.
    pub(super) fn compare_floats(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: CompareOp,
        left: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<JitValue> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        let (l, r) = (left.data, right.data);
        let equal = builder.ins().fcmp(FloatCC::Equal, l, r);
        let left_nan = builder.ins().fcmp(FloatCC::Unordered, l, l);
        let right_nan = builder.ins().fcmp(FloatCC::Unordered, r, r);
        let both_nan = builder.ins().band(left_nan, right_nan);
        let total_equal = builder.ins().bor(equal, both_nan);
        let right_not_nan = builder.ins().bxor_imm_u(right_nan, 1);
        let left_nan_greater = builder.ins().band(left_nan, right_not_nan);
        let greater = builder.ins().fcmp(FloatCC::GreaterThan, l, r);
        let total_greater = builder.ins().bor(greater, left_nan_greater);
        let left_not_nan = builder.ins().bxor_imm_u(left_nan, 1);
        let right_nan_greater = builder.ins().band(right_nan, left_not_nan);
        let less = builder.ins().fcmp(FloatCC::LessThan, l, r);
        let total_less = builder.ins().bor(less, right_nan_greater);
        let data = match op {
            CompareOp::Equal | CompareOp::EqualMissing => total_equal,
            CompareOp::NotEqual | CompareOp::NotEqualMissing => {
                builder.ins().bxor_imm_u(total_equal, 1)
            },
            CompareOp::LessThan => total_less,
            CompareOp::LessThanOrEqual => builder.ins().bor(total_less, total_equal),
            CompareOp::GreaterThan => total_greater,
            CompareOp::GreaterThanOrEqual => builder.ins().bor(total_greater, total_equal),
        };
        Ok(comparison_result(builder, op, data, left, right))
    }

    /// Compare integers or Booleans, signed or unsigned according to the operands' type.
    pub(super) fn compare_integers(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: CompareOp,
        left_program: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<JitValue> {
        let left = self.emit(builder, left_program)?;
        let right = self.emit(builder, right)?;
        let condition = match (op, left_program.dtype().is_signed()) {
            (CompareOp::Equal | CompareOp::EqualMissing, _) => IntCC::Equal,
            (CompareOp::NotEqual | CompareOp::NotEqualMissing, _) => IntCC::NotEqual,
            (CompareOp::LessThan, true) => IntCC::SignedLessThan,
            (CompareOp::LessThan, false) => IntCC::UnsignedLessThan,
            (CompareOp::LessThanOrEqual, true) => IntCC::SignedLessThanOrEqual,
            (CompareOp::LessThanOrEqual, false) => IntCC::UnsignedLessThanOrEqual,
            (CompareOp::GreaterThan, true) => IntCC::SignedGreaterThan,
            (CompareOp::GreaterThan, false) => IntCC::UnsignedGreaterThan,
            (CompareOp::GreaterThanOrEqual, true) => IntCC::SignedGreaterThanOrEqual,
            (CompareOp::GreaterThanOrEqual, false) => IntCC::UnsignedGreaterThanOrEqual,
        };
        let data = builder.ins().icmp(condition, left.data, right.data);
        Ok(comparison_result(builder, op, data, left, right))
    }

    /// Boolean `and`/`or` with Polars' Kleene logic: `false and null` is `false` and
    /// `true or null` is `true`; otherwise a null operand makes the result null.
    pub(super) fn logical(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: BooleanOp,
        left: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<JitValue> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        let left_true = builder.ins().band(left.data, left.valid);
        let right_true = builder.ins().band(right.data, right.valid);
        let left_false_data = builder.ins().bxor_imm_u(left.data, 1);
        let right_false_data = builder.ins().bxor_imm_u(right.data, 1);
        let left_false = builder.ins().band(left_false_data, left.valid);
        let right_false = builder.ins().band(right_false_data, right.valid);
        let both_valid = builder.ins().band(left.valid, right.valid);
        Ok(match op {
            BooleanOp::And => JitValue {
                data: builder.ins().band(left_true, right_true),
                valid: {
                    let either_false = builder.ins().bor(left_false, right_false);
                    builder.ins().bor(either_false, both_valid)
                },
            },
            BooleanOp::Or => JitValue {
                data: builder.ins().bor(left_true, right_true),
                valid: {
                    let either_true = builder.ins().bor(left_true, right_true);
                    builder.ins().bor(either_true, both_valid)
                },
            },
        })
    }
}

/// A comparison's value given the operands' comparison `data`. Ordinary comparisons are null
/// when either operand is null; `eq_missing`/`ne_missing` treat two nulls as equal and are never
/// null.
fn comparison_result(
    builder: &mut FunctionBuilder<'_>,
    op: CompareOp,
    data: Value,
    left: JitValue,
    right: JitValue,
) -> JitValue {
    let both_valid = builder.ins().band(left.valid, right.valid);
    if !matches!(op, CompareOp::EqualMissing | CompareOp::NotEqualMissing) {
        return JitValue {
            data,
            valid: both_valid,
        };
    }
    let equal_data = if matches!(op, CompareOp::NotEqualMissing) {
        builder.ins().bxor_imm_u(data, 1)
    } else {
        data
    };
    let compared = builder.ins().band(both_valid, equal_data);
    let left_null = builder.ins().bxor_imm_u(left.valid, 1);
    let right_null = builder.ins().bxor_imm_u(right.valid, 1);
    let both_null = builder.ins().band(left_null, right_null);
    let equal_missing = builder.ins().bor(compared, both_null);
    JitValue {
        data: if matches!(op, CompareOp::NotEqualMissing) {
            builder.ins().bxor_imm_u(equal_missing, 1)
        } else {
            equal_missing
        },
        valid: builder.ins().iconst(types::I8, 1),
    }
}
