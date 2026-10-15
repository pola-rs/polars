//! Small helpers shared across compilation.

use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Type, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::{PolarsError, polars_err};

use crate::jit_program::ScalarType;

/// The machine type of a one-word value of `dtype`.
pub(super) fn cranelift_type(dtype: ScalarType) -> Type {
    match dtype {
        ScalarType::Boolean => types::I8,
        ScalarType::Int32 | ScalarType::UInt32 => types::I32,
        ScalarType::Int64 | ScalarType::UInt64 => types::I64,
        ScalarType::Float32 => types::F32,
        ScalarType::Float64 => types::F64,
        ScalarType::String => unreachable!("strings take two words; see `value_types`"),
    }
}

/// A float constant of type `dtype` (`F32` or `F64`).
pub(super) fn float_constant(builder: &mut FunctionBuilder<'_>, dtype: Type, value: f64) -> Value {
    if dtype == types::F32 {
        let bits = (value as f32).to_bits();
        let integer = builder.ins().iconst(types::I32, i64::from(bits));
        builder
            .ins()
            .bitcast(types::F32, MemFlagsData::new(), integer)
    } else {
        let integer = builder.ins().iconst(types::I64, value.to_bits() as i64);
        builder
            .ins()
            .bitcast(types::F64, MemFlagsData::new(), integer)
    }
}

pub(super) fn compile_error(error: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "could not compile JIT program: {error}")
}
