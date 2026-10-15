//! Casts, with Polars' strict, non-strict, and wrapping overflow rules.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::{PolarsResult, polars_err};

use super::Emitter;
use crate::compiler::helpers::{cranelift_type, float_constant};
use crate::compiler::values::JitValue;
use crate::jit_program::{CastOverflow, JitProgram, ScalarType};

impl Emitter<'_> {
    pub(super) fn cast(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        input: &JitProgram,
        dtype: ScalarType,
        overflow: CastOverflow,
    ) -> PolarsResult<JitValue> {
        let value = self.emit(builder, input)?;
        let source = input.dtype();
        let valid = match overflow {
            CastOverflow::Unchecked => value.valid,
            CastOverflow::Null => {
                let fits = fits_target(builder, value.data, source, dtype)?;
                builder.ins().band(value.valid, fits)
            },
            CastOverflow::Raise => {
                // A valid value that does not fit stops the fold; a null input stays null.
                let fits = fits_target(builder, value.data, source, dtype)?;
                let does_not_fit = builder.ins().bxor_imm_u(fits, 1);
                let failed = builder.ins().band(value.valid, does_not_fit);
                let report = builder.create_block();
                let next = builder.create_block();
                builder.ins().brif(failed, report, &[], next, &[]);

                // Record which cast failed and on which value, then leave the loop.
                builder.switch_to_block(report);
                let cast_id = self.strict_casts.len() as i64;
                let cast_id = builder.ins().iconst(types::I64, cast_id);
                let bits = raw_bits_u64(builder, value.data, source);
                let flags = MemFlagsData::new();
                builder.ins().store(flags, cast_id, self.cast_failure, 0);
                builder.ins().store(flags, bits, self.cast_failure, 8);
                builder.ins().jump(self.failure, &[self.index.into()]);

                builder.switch_to_block(next);
                self.strict_casts.push((source, dtype));
                value.valid
            },
        };
        Ok(JitValue {
            data: convert(builder, value.data, source, dtype)?,
            valid,
        })
    }
}

/// Convert a primitive from `source` to `target`. Out-of-range values wrap or saturate;
/// `fits_target` reports whether they fit.
fn convert(
    builder: &mut FunctionBuilder<'_>,
    value: Value,
    source: ScalarType,
    target: ScalarType,
) -> PolarsResult<Value> {
    if source == target {
        return Ok(value);
    }
    let source_type = cranelift_type(source);
    let target_type = cranelift_type(target);
    if target == ScalarType::Boolean {
        // Polars maps every nonzero value, including NaN, to true.
        return Ok(if source.is_float() {
            let zero = float_constant(builder, source_type, 0.0);
            builder.ins().fcmp(FloatCC::NotEqual, value, zero)
        } else {
            builder.ins().icmp_imm_s(IntCC::NotEqual, value, 0)
        });
    }
    if !source.is_float() && !target.is_float() {
        return Ok(match source_type.bits().cmp(&target_type.bits()) {
            std::cmp::Ordering::Less if source.is_signed() => {
                builder.ins().sextend(target_type, value)
            },
            std::cmp::Ordering::Less => builder.ins().uextend(target_type, value),
            std::cmp::Ordering::Greater => builder.ins().ireduce(target_type, value),
            std::cmp::Ordering::Equal => value,
        });
    }
    if source.is_float() && target.is_float() {
        return Ok(if source_type.bits() < target_type.bits() {
            builder.ins().fpromote(target_type, value)
        } else {
            builder.ins().fdemote(target_type, value)
        });
    }
    if !source.is_float() && target.is_float() {
        return Ok(if source.is_signed() {
            builder.ins().fcvt_from_sint(target_type, value)
        } else {
            builder.ins().fcvt_from_uint(target_type, value)
        });
    }
    if source.is_float() && !target.is_float() && target != ScalarType::Boolean {
        return Ok(if target.is_signed() {
            builder.ins().fcvt_to_sint_sat(target_type, value)
        } else {
            builder.ins().fcvt_to_uint_sat(target_type, value)
        });
    }
    Err(polars_err!(InvalidOperation:
        "unsupported JIT cast from {} to {}", source.dtype(), target.dtype()))
}

/// Whether a primitive of `source` has a value in `target`, as `0` or `1`.
fn fits_target(
    builder: &mut FunctionBuilder<'_>,
    value: Value,
    source: ScalarType,
    target: ScalarType,
) -> PolarsResult<Value> {
    if source.is_float() && !target.is_float() && target != ScalarType::Boolean {
        let float_type = cranelift_type(source);
        let lower = if target.is_signed() {
            -(2_f64.powi(i32::from(target.bit_width()) - 1))
        } else {
            0.0
        };
        let upper = if target.is_signed() {
            2_f64.powi(i32::from(target.bit_width()) - 1)
        } else {
            2_f64.powi(i32::from(target.bit_width()))
        };
        let lower = float_constant(builder, float_type, lower);
        let upper = float_constant(builder, float_type, upper);
        // Polars truncates toward zero before checking the range, so -0.5 casts to 0_u32.
        let truncated = builder.ins().trunc(value);
        let at_least_lower = builder
            .ins()
            .fcmp(FloatCC::GreaterThanOrEqual, truncated, lower);
        let below_upper = builder.ins().fcmp(FloatCC::LessThan, truncated, upper);
        return Ok(builder.ins().band(at_least_lower, below_upper));
    }
    if !source.is_float() && !target.is_float() && target != ScalarType::Boolean {
        let cast = convert(builder, value, source, target)?;
        let round_trip = convert(builder, cast, target, source)?;
        let same = builder.ins().icmp(IntCC::Equal, value, round_trip);
        if source.is_signed() && !target.is_signed() {
            let non_negative = builder
                .ins()
                .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, value, 0);
            return Ok(builder.ins().band(same, non_negative));
        }
        if !source.is_signed() && target.is_signed() {
            // The round trip preserves the bits of values above the signed maximum, such as
            // 2^63_u64 as i64, so the signed result must also be non-negative.
            let non_negative = builder
                .ins()
                .icmp_imm_s(IntCC::SignedGreaterThanOrEqual, cast, 0);
            return Ok(builder.ins().band(same, non_negative));
        }
        return Ok(same);
    }
    Err(polars_err!(InvalidOperation:
        "unsupported nullable JIT cast from {} to {}", source.dtype(), target.dtype()))
}

/// Reinterpret a primitive as its raw bits, zero-extended to 64 bits.
fn raw_bits_u64(builder: &mut FunctionBuilder<'_>, value: Value, dtype: ScalarType) -> Value {
    let value_type = cranelift_type(dtype);
    let bits = if dtype.is_float() {
        builder
            .ins()
            .bitcast(value_type.as_int(), MemFlagsData::new(), value)
    } else {
        value
    };
    if value_type.bits() < 64 {
        builder.ins().uextend(types::I64, bits)
    } else {
        bits
    }
}
