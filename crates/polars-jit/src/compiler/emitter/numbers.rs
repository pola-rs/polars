//! Numeric instructions and calls to registered runtime functions.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_defs::expr::RoundMode;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::helpers::{cranelift_type, float_constant};
use crate::compiler::runtime_functions::RuntimeFunction;
use crate::compiler::values::{JitData, JitValue};
use crate::jit_program::{JitProgram, NumericUnary, ScalarType};

impl Emitter<'_> {
    pub(super) fn numeric_unary(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        op: NumericUnary,
        input: &JitProgram,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let value = self.emit(builder, input)?;
        let number = value.data.scalar();
        let ty = cranelift_type(dtype);
        let data = if dtype.is_float() {
            match op {
                NumericUnary::Negate => builder.ins().fneg(number),
                NumericUnary::Abs => builder.ins().fabs(number),
                NumericUnary::Sqrt => builder.ins().sqrt(number),
                NumericUnary::Floor => builder.ins().floor(number),
                NumericUnary::Ceil => builder.ins().ceil(number),
                NumericUnary::Sign => {
                    let zero = float_constant(builder, ty, 0.0);
                    let one = float_constant(builder, ty, 1.0);
                    let minus_one = float_constant(builder, ty, -1.0);
                    let negative = builder.ins().fcmp(FloatCC::LessThan, number, zero);
                    let positive = builder.ins().fcmp(FloatCC::GreaterThan, number, zero);
                    let nonnegative = builder.ins().select(positive, one, number);
                    builder.ins().select(negative, minus_one, nonnegative)
                },
                NumericUnary::Round {
                    decimals: 0,
                    mode: RoundMode::HalfToEven,
                } => builder.ins().nearest(number),
                NumericUnary::Round { decimals, mode } => {
                    let function = if dtype == ScalarType::Float32 {
                        RuntimeFunction::RoundF32
                    } else {
                        RuntimeFunction::RoundF64
                    };
                    let decimals = builder.ins().iconst(types::I32, i64::from(decimals));
                    let mode = builder.ins().iconst(
                        types::I32,
                        match mode {
                            RoundMode::HalfToEven => 0,
                            RoundMode::HalfAwayFromZero => 1,
                            RoundMode::ToZero => 2,
                        },
                    );
                    self.call(builder, function, &[number, decimals, mode])
                        .unwrap()
                },
            }
        } else {
            match op {
                NumericUnary::Negate => builder.ins().ineg(number),
                NumericUnary::Abs if dtype.is_signed() => builder.ins().iabs(number),
                NumericUnary::Sign => {
                    let zero = builder.ins().iconst(ty, 0);
                    let one = builder.ins().iconst(ty, 1);
                    let nonzero = builder.ins().icmp_imm_s(IntCC::NotEqual, number, 0);
                    let positive = builder.ins().select(nonzero, one, zero);
                    if dtype.is_signed() {
                        let negative = builder.ins().icmp_imm_s(IntCC::SignedLessThan, number, 0);
                        let minus_one = builder.ins().iconst(ty, -1);
                        builder.ins().select(negative, minus_one, positive)
                    } else {
                        positive
                    }
                },
                NumericUnary::Sqrt => unreachable!("sqrt input has been cast to float"),
                _ => number,
            }
        };
        Ok(JitValue {
            data: JitData::Scalar(data),
            valid: value.valid,
        })
    }

    pub(super) fn pow(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        base: &JitProgram,
        exponent: &JitProgram,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let base = self.emit(builder, base)?;
        let exponent = self.emit(builder, exponent)?;
        let mut number = base.data.scalar();
        let function = match dtype {
            ScalarType::Float32 => RuntimeFunction::PowF32,
            ScalarType::Float64 => RuntimeFunction::PowF64,
            _ => {
                if dtype.bit_width() == 32 {
                    number = builder.ins().uextend(types::I64, number);
                }
                RuntimeFunction::PowInteger
            },
        };
        let mut data = self
            .call(builder, function, &[number, exponent.data.scalar()])
            .unwrap();
        if !dtype.is_float() && dtype.bit_width() == 32 {
            data = builder.ins().ireduce(types::I32, data);
        }
        Ok(JitValue {
            data: JitData::Scalar(data),
            valid: builder.ins().band(base.valid, exponent.valid),
        })
    }

    pub(super) fn clip(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        input: &JitProgram,
        min: Option<&JitProgram>,
        max: Option<&JitProgram>,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let value = self.emit(builder, input)?;
        // Evaluate both bounds (including their strict casts) before choosing the result.
        let min = min.map(|bound| self.emit(builder, bound)).transpose()?;
        let max = max.map(|bound| self.emit(builder, bound)).transpose()?;
        let number = value.data.scalar();
        let mut data = number;
        if let Some(max) = max {
            let above = compare(builder, dtype, true, number, max.data.scalar());
            let above = builder.ins().band(above, max.valid);
            data = builder.ins().select(above, max.data.scalar(), data);
        }
        if let Some(min) = min {
            let below = compare(builder, dtype, false, number, min.data.scalar());
            let below = builder.ins().band(below, min.valid);
            data = builder.ins().select(below, min.data.scalar(), data);
        }
        Ok(JitValue {
            data: JitData::Scalar(data),
            valid: value.valid,
        })
    }

    pub(super) fn min_max(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        left: &JitProgram,
        right: &JitProgram,
        max: bool,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let left = self.emit(builder, left)?;
        let right = self.emit(builder, right)?;
        let (l, r) = (left.data.scalar(), right.data.scalar());
        let mut better = compare(builder, dtype, max, l, r);
        if dtype.is_float() {
            let right_nan = builder.ins().fcmp(FloatCC::Unordered, r, r);
            better = builder.ins().bor(better, right_nan);
        }
        let right_null = builder.ins().bxor_imm_u(right.valid, 1);
        let choose_left = builder.ins().bor(better, right_null);
        let choose_left = builder.ins().band(choose_left, left.valid);
        Ok(JitValue {
            data: JitData::Scalar(builder.ins().select(choose_left, l, r)),
            valid: builder.ins().bor(left.valid, right.valid),
        })
    }
}

/// Ordinary partial ordering for floats, so NaN comparisons are false (as clip requires).
fn compare(
    builder: &mut FunctionBuilder<'_>,
    dtype: ScalarType,
    greater: bool,
    left: Value,
    right: Value,
) -> Value {
    if dtype.is_float() {
        builder.ins().fcmp(
            if greater {
                FloatCC::GreaterThan
            } else {
                FloatCC::LessThan
            },
            left,
            right,
        )
    } else {
        let condition = match (dtype.is_signed(), greater) {
            (true, true) => IntCC::SignedGreaterThan,
            (true, false) => IntCC::SignedLessThan,
            (false, true) => IntCC::UnsignedGreaterThan,
            (false, false) => IntCC::UnsignedLessThan,
        };
        builder.ins().icmp(condition, left, right)
    }
}
