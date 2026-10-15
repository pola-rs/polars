//! Literals and nulls.

use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::helpers::cranelift_type;
use crate::compiler::values::JitValue;
use crate::jit_program::{JitLiteral, ScalarType};

impl Emitter<'_> {
    pub(super) fn literal(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        literal: &JitLiteral,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let bits = match literal {
            JitLiteral::Boolean(v) => u128::from(*v),
            JitLiteral::Signed(v) => *v as u128,
            JitLiteral::Unsigned(v) => *v,
            JitLiteral::Float32(v) => u128::from(*v),
            JitLiteral::Float64(v) => u128::from(*v),
        };
        let integer_type = cranelift_type(dtype).as_int();
        let integer = builder.ins().iconst(integer_type, bits as u64 as i64);
        let data = if dtype.is_float() {
            builder
                .ins()
                .bitcast(cranelift_type(dtype), MemFlagsData::new(), integer)
        } else {
            integer
        };
        Ok(JitValue {
            data,
            valid: builder.ins().iconst(types::I8, 1),
        })
    }

    /// A null of `dtype`.
    pub(super) fn null(&self, builder: &mut FunctionBuilder<'_>, dtype: ScalarType) -> JitValue {
        JitValue {
            data: zero(builder, dtype),
            valid: builder.ins().iconst(types::I8, 0),
        }
    }
}

/// The zero value of `dtype`.
fn zero(builder: &mut FunctionBuilder<'_>, dtype: ScalarType) -> Value {
    let integer = builder.ins().iconst(cranelift_type(dtype).as_int(), 0);
    if dtype.is_float() {
        builder
            .ins()
            .bitcast(cranelift_type(dtype), MemFlagsData::new(), integer)
    } else {
        integer
    }
}
