//! Functions: `not`, the null checks, the NaN and infinity checks, string lengths, and
//! `fill_null`.

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{InstBuilder, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::helpers::{cranelift_type, float_constant};
use crate::compiler::runtime_functions::RuntimeFunction;
use crate::compiler::values::{
    JitData, JitValue, append_value_params, value_args, value_from_params,
};
use crate::jit_program::{FloatCheck, JitProgram, ScalarType};

impl Emitter<'_> {
    /// `not`; null stays null.
    pub(super) fn not(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        input: &JitProgram,
    ) -> PolarsResult<JitValue> {
        let value = self.emit(builder, input)?;
        Ok(JitValue {
            data: JitData::Scalar(builder.ins().bxor_imm_u(value.data.scalar(), 1)),
            valid: value.valid,
        })
    }

    /// `is_null`, which is never null itself.
    pub(super) fn is_null(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        input: &JitProgram,
    ) -> PolarsResult<JitValue> {
        let value = self.emit(builder, input)?;
        Ok(JitValue {
            data: JitData::Scalar(builder.ins().bxor_imm_u(value.valid, 1)),
            valid: builder.ins().iconst(types::I8, 1),
        })
    }

    /// `is_nan`, `is_finite`, or `is_infinite`. Integers are never NaN or infinite, and the
    /// result is null exactly when the input is null.
    pub(super) fn float_check(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        check: FloatCheck,
        input: &JitProgram,
    ) -> PolarsResult<JitValue> {
        let value = self.emit(builder, input)?;
        let data = if input.dtype().is_float() {
            let value_type = cranelift_type(input.dtype());
            let number = value.data.scalar();
            match check {
                // NaN is the only value that is unordered with itself.
                FloatCheck::Nan => builder.ins().fcmp(FloatCC::Unordered, number, number),
                FloatCheck::Finite | FloatCheck::Infinite => {
                    let magnitude = builder.ins().fabs(number);
                    let infinity = float_constant(builder, value_type, f64::INFINITY);
                    // Both comparisons are false for NaN, which is neither finite nor infinite.
                    let condition = if check == FloatCheck::Finite {
                        FloatCC::LessThan
                    } else {
                        FloatCC::Equal
                    };
                    builder.ins().fcmp(condition, magnitude, infinity)
                },
            }
        } else {
            let finite = check == FloatCheck::Finite;
            builder.ins().iconst(types::I8, i64::from(finite))
        };
        Ok(JitValue {
            data: JitData::Scalar(data),
            valid: value.valid,
        })
    }

    /// `str.len_bytes()`, or `str.len_chars()` when `chars` is set, as UInt32. The result is
    /// null exactly when the input is null.
    pub(super) fn string_length(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        input: &JitProgram,
        chars: bool,
    ) -> PolarsResult<JitValue> {
        let value = self.emit(builder, input)?;
        let (ptr, len) = value.data.string();
        let length = if chars {
            self.call(builder, RuntimeFunction::LenChars, &[ptr, len])
                .expect("len_chars returns a length")
        } else {
            // Polars string views are at most u32::MAX bytes long.
            builder.ins().ireduce(types::I32, len)
        };
        Ok(JitValue {
            data: JitData::Scalar(length),
            valid: value.valid,
        })
    }

    /// `value`, or `fill` where it is null. The value is evaluated once, and the fill only where
    /// the value is null.
    pub(super) fn fill_null(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        value: &JitProgram,
        fill: &JitProgram,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let value = self.emit(builder, value)?;
        let fill_block = builder.create_block();
        let merge = builder.create_block();
        append_value_params(builder, merge, dtype);
        builder
            .ins()
            .brif(value.valid, merge, &value_args(value), fill_block, &[]);
        builder.switch_to_block(fill_block);
        let fill = self.emit(builder, fill)?;
        builder.ins().jump(merge, &value_args(fill));
        builder.switch_to_block(merge);
        Ok(value_from_params(builder.block_params(merge), dtype))
    }
}
