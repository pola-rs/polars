//! The `Emitter` generates Cranelift IR for one fold body inside the row loop.
//!
//! `emit` is the single place that matches `JitProgram` nodes: every arm calls the method that
//! generates that node. The methods live in the modules below, grouped by kind like the
//! translator's.

mod operators;

use cranelift_codegen::ir::Value;
use cranelift_frontend::FunctionBuilder;
use polars_error::{PolarsResult, polars_err};

use crate::jit_program::{InputId, JitProgram};

/// State shared while emitting one fold body.
pub(super) struct Emitter<'a> {
    /// The value of every input the body can read in the current row.
    inputs: &'a [(InputId, Value)],
}

impl<'a> Emitter<'a> {
    pub(super) fn new(inputs: &'a [(InputId, Value)]) -> Self {
        Self { inputs }
    }

    /// Generate `program` at the builder's current position and return its value.
    pub(super) fn emit(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        program: &JitProgram,
    ) -> PolarsResult<Value> {
        match program {
            JitProgram::Input { input, .. } => self.input(*input),

            // Operators, in `operators`.
            JitProgram::Binary {
                op, left, right, ..
            } => self.arithmetic(builder, *op, left, right),
        }
    }

    /// The carried state or a row input, already loaded for the current row.
    fn input(&self, input: InputId) -> PolarsResult<Value> {
        self.inputs
            .iter()
            .find(|(candidate, _)| *candidate == input)
            .map(|(_, value)| *value)
            .ok_or_else(|| polars_err!(InvalidOperation: "missing JIT input {input:?}"))
    }
}
