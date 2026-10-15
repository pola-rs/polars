//! Compilation of a typed `JitProgram` into native code with Cranelift.
//!
//! `list_program` generates the function that loops over the rows, and the `Emitter`
//! generates the fold body inside it.

mod emitter;
mod helpers;
mod list_program;
mod runtime_functions;
mod types;
mod values;

use polars_error::PolarsResult;
pub(crate) use runtime_functions::StringContext;
pub use types::CompiledProgram;
pub(crate) use types::{CastFailure, CompilationShape, InputBinding, ListBindings};

use crate::jit_program::JitProgram;

pub(crate) fn compile(
    program: &JitProgram,
    shape: CompilationShape,
) -> PolarsResult<CompiledProgram> {
    match shape {
        CompilationShape::Lists(bindings) => list_program::compile(program, &bindings),
    }
}
