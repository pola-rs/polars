//! Casts between the core scalar types.

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use polars_error::{PolarsResult, polars_err};

use super::Emitter;
use crate::compiler::helpers::cranelift_type;
use crate::jit_program::{JitProgram, ScalarType};

impl Emitter<'_> {
    pub(super) fn cast(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        input: &JitProgram,
        dtype: ScalarType,
    ) -> PolarsResult<Value> {
        let value = self.emit(builder, input)?;
        convert(builder, value, input.dtype(), dtype)
    }
}

/// Convert a value from `source` to `target`.
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
    Err(polars_err!(InvalidOperation:
        "unsupported JIT cast from {} to {}", source.dtype(), target.dtype()))
}
