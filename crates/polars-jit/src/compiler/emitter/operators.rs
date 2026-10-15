//! Binary operators: arithmetic, string concatenation, comparisons, and Boolean `and`/`or`.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::runtime_functions::RuntimeFunction;
use crate::compiler::values::{JitData, JitValue};
use crate::jit_program::{BinaryOp, BooleanOp, CompareOp, JitProgram, ScalarType};

impl Emitter<'_> {
    /// Numeric arithmetic; null when either operand is null or an integer divisor is zero.
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
        let (l, r) = (left.data.scalar(), right.data.scalar());
        let mut valid = builder.ins().band(left.valid, right.valid);
        let data = if dtype.is_float() {
            match op {
                BinaryOp::Add => builder.ins().fadd(l, r),
                BinaryOp::Subtract => builder.ins().fsub(l, r),
                BinaryOp::Multiply => builder.ins().fmul(l, r),
                BinaryOp::TrueDivide => builder.ins().fdiv(l, r),
                BinaryOp::FloorDivide | BinaryOp::Modulus => {
                    let quotient = builder.ins().fdiv(l, r);
                    let floor = builder.ins().floor(quotient);
                    if op == BinaryOp::FloorDivide {
                        floor
                    } else {
                        // Match Polars' floating modulo, including infinities and signed zero.
                        let product = builder.ins().fmul(r, floor);
                        builder.ins().fsub(l, product)
                    }
                },
            }
        } else {
            match op {
                BinaryOp::Add => builder.ins().iadd(l, r),
                BinaryOp::Subtract => builder.ins().isub(l, r),
                BinaryOp::Multiply => builder.ins().imul(l, r),
                BinaryOp::FloorDivide | BinaryOp::Modulus => {
                    let nonzero = builder.ins().icmp_imm_s(IntCC::NotEqual, r, 0);
                    valid = builder.ins().band(valid, nonzero);
                    integer_division(builder, op, dtype, l, r, nonzero)
                },
                BinaryOp::TrueDivide => unreachable!("true division has a floating result"),
            }
        };
        Ok(JitValue {
            data: JitData::Scalar(data),
            valid,
        })
    }

    /// `+` on strings, which concatenates them in the row's scratch storage.
    pub(super) fn concat(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        left: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<JitValue> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        let (left_ptr, left_len) = left.data.string();
        let (right_ptr, right_len) = right.data.string();
        let strings = self.strings;
        let ptr = self
            .call(
                builder,
                RuntimeFunction::Concat,
                &[strings, left_ptr, left_len, right_ptr, right_len],
            )
            .expect("concat returns a pointer");
        self.allocates_strings = true;
        Ok(JitValue {
            data: JitData::Str {
                ptr,
                len: builder.ins().iadd(left_len, right_len),
            },
            valid: builder.ins().band(left.valid, right.valid),
        })
    }

    /// Compare strings by their UTF-8 bytes through a runtime function.
    pub(super) fn compare_strings(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: CompareOp,
        left: &JitProgram,
        right: &JitProgram,
    ) -> PolarsResult<JitValue> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        let (left_ptr, left_len) = left.data.string();
        let (right_ptr, right_len) = right.data.string();
        // The runtime function returns -1, 0, or 1.
        let order = self
            .call(
                builder,
                RuntimeFunction::Compare,
                &[left_ptr, left_len, right_ptr, right_len],
            )
            .expect("compare returns an ordering");
        let condition = match op {
            CompareOp::Equal | CompareOp::EqualMissing => IntCC::Equal,
            CompareOp::NotEqual | CompareOp::NotEqualMissing => IntCC::NotEqual,
            CompareOp::LessThan => IntCC::SignedLessThan,
            CompareOp::LessThanOrEqual => IntCC::SignedLessThanOrEqual,
            CompareOp::GreaterThan => IntCC::SignedGreaterThan,
            CompareOp::GreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        };
        let data = builder.ins().icmp_imm_s(condition, order, 0);
        Ok(comparison_result(builder, op, data, left, right))
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
        let (l, r) = (left.data.scalar(), right.data.scalar());
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
        let data = builder
            .ins()
            .icmp(condition, left.data.scalar(), right.data.scalar());
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
        let left_true = builder.ins().band(left.data.scalar(), left.valid);
        let right_true = builder.ins().band(right.data.scalar(), right.valid);
        let left_false_data = builder.ins().bxor_imm_u(left.data.scalar(), 1);
        let right_false_data = builder.ins().bxor_imm_u(right.data.scalar(), 1);
        let left_false = builder.ins().band(left_false_data, left.valid);
        let right_false = builder.ins().band(right_false_data, right.valid);
        let both_valid = builder.ins().band(left.valid, right.valid);
        Ok(match op {
            BooleanOp::And => JitValue {
                data: JitData::Scalar(builder.ins().band(left_true, right_true)),
                valid: {
                    let either_false = builder.ins().bor(left_false, right_false);
                    builder.ins().bor(either_false, both_valid)
                },
            },
            BooleanOp::Or => JitValue {
                data: JitData::Scalar(builder.ins().bor(left_true, right_true)),
                valid: {
                    let either_true = builder.ins().bor(left_true, right_true);
                    builder.ins().bor(either_true, both_valid)
                },
            },
        })
    }
}

/// Replace trapping divisors before emitting a machine division, including for null operands.
/// Dividing MIN by 1 yields the required wrapping result for MIN / -1, with remainder zero.
fn integer_division(
    builder: &mut FunctionBuilder<'_>,
    op: BinaryOp,
    dtype: ScalarType,
    left: Value,
    right: Value,
    nonzero: Value,
) -> Value {
    let ty = builder.func.dfg.value_type(left);
    let one = builder.ins().iconst(ty, 1);
    let mut safe = nonzero;
    if dtype.is_signed() {
        let minimum = if dtype == ScalarType::Int32 {
            i64::from(i32::MIN)
        } else {
            i64::MIN
        };
        let is_minimum = builder.ins().icmp_imm_s(IntCC::Equal, left, minimum);
        let is_minus_one = builder.ins().icmp_imm_s(IntCC::Equal, right, -1);
        let overflow = builder.ins().band(is_minimum, is_minus_one);
        let no_overflow = builder.ins().bxor_imm_u(overflow, 1);
        safe = builder.ins().band(safe, no_overflow);
    }
    let divisor = builder.ins().select(safe, right, one);
    if !dtype.is_signed() {
        return if op == BinaryOp::Modulus {
            builder.ins().urem(left, divisor)
        } else {
            builder.ins().udiv(left, divisor)
        };
    }
    let remainder = builder.ins().srem(left, divisor);
    let has_remainder = builder.ins().icmp_imm_s(IntCC::NotEqual, remainder, 0);
    let signs = builder.ins().bxor(left, right);
    let opposite_signs = builder.ins().icmp_imm_s(IntCC::SignedLessThan, signs, 0);
    let correct = builder.ins().band(has_remainder, opposite_signs);
    if op == BinaryOp::Modulus {
        let zero = builder.ins().iconst(ty, 0);
        let correction = builder.ins().select(correct, right, zero);
        builder.ins().iadd(remainder, correction)
    } else {
        let quotient = builder.ins().sdiv(left, divisor);
        let correction = builder.ins().uextend(ty, correct);
        builder.ins().isub(quotient, correction)
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
            data: JitData::Scalar(data),
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
        data: JitData::Scalar(if matches!(op, CompareOp::NotEqualMissing) {
            builder.ins().bxor_imm_u(equal_missing, 1)
        } else {
            equal_missing
        }),
        valid: builder.ins().iconst(types::I8, 1),
    }
}
