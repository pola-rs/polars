//! Literals and nulls.

use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::helpers::cranelift_type;
use crate::compiler::values::{JitData, JitValue};
use crate::jit_program::{JitLiteral, ScalarType};

impl Emitter<'_> {
    pub(super) fn literal(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        literal: &JitLiteral,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let data = if let JitLiteral::String(value) = literal {
            // The compiled program owns the bytes, so the embedded pointer outlives every call.
            let bytes: Box<[u8]> = value.as_bytes().into();
            let ptr = builder.ins().iconst(types::I64, bytes.as_ptr() as i64);
            let len = builder.ins().iconst(types::I64, bytes.len() as i64);
            self.string_literals.push(bytes);
            JitData::Str { ptr, len }
        } else {
            let bits = match literal {
                JitLiteral::String(_) => unreachable!("handled above"),
                JitLiteral::Boolean(v) => u128::from(*v),
                JitLiteral::Signed(v) => *v as u128,
                JitLiteral::Unsigned(v) => *v,
                JitLiteral::Float32(v) => u128::from(*v),
                JitLiteral::Float64(v) => u128::from(*v),
            };
            let integer_type = cranelift_type(dtype).as_int();
            let integer = builder.ins().iconst(integer_type, bits as u64 as i64);
            JitData::Scalar(if dtype.is_float() {
                builder
                    .ins()
                    .bitcast(cranelift_type(dtype), MemFlagsData::new(), integer)
            } else {
                integer
            })
        };
        Ok(JitValue {
            data,
            valid: builder.ins().iconst(types::I8, 1),
        })
    }

    /// A null of `dtype`. A null string is empty, so runtime functions never read through it.
    pub(super) fn null(&self, builder: &mut FunctionBuilder<'_>, dtype: ScalarType) -> JitValue {
        let data = match dtype {
            ScalarType::String => JitData::Str {
                ptr: builder.ins().iconst(types::I64, 0),
                len: builder.ins().iconst(types::I64, 0),
            },
            dtype => JitData::Scalar(zero(builder, dtype)),
        };
        JitValue {
            data,
            valid: builder.ins().iconst(types::I8, 0),
        }
    }
}

/// The zero value of a primitive `dtype`.
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
