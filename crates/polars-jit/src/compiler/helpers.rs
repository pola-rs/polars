//! Small helpers shared across compilation.

use cranelift_codegen::ir::{Type, types};
use polars_error::{PolarsError, polars_err};

use crate::jit_program::ScalarType;

/// The machine type of a value of `dtype`.
pub(super) fn cranelift_type(dtype: ScalarType) -> Type {
    match dtype {
        ScalarType::Boolean => types::I8,
        ScalarType::Int32 | ScalarType::UInt32 => types::I32,
        ScalarType::Int64 | ScalarType::UInt64 => types::I64,
        ScalarType::Float32 => types::F32,
        ScalarType::Float64 => types::F64,
    }
}

pub(super) fn compile_error(error: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "could not compile JIT program: {error}")
}
