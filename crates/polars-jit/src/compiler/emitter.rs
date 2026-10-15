//! The `Emitter` generates Cranelift IR for one fold body inside the row loop.
//!
//! `emit` is the single place that matches `JitProgram` nodes: every arm calls the method that
//! generates that node. The methods live in the modules below, grouped by kind like the
//! translator's.

mod casts;
mod conditionals;
mod functions;
mod literals;
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
            JitProgram::Literal { value, dtype } => self.literal(builder, value, *dtype),
            JitProgram::Cast { input, dtype } => self.cast(builder, input, *dtype),

            // Operators, in `operators`.
            JitProgram::Binary {
                op,
                left,
                right,
                dtype,
            } => self.arithmetic(builder, *op, left, right, *dtype),
            JitProgram::Compare { op, left, right } if left.dtype().is_float() => {
                self.compare_floats(builder, *op, left, right)
            },
            JitProgram::Compare { op, left, right } => {
                self.compare_integers(builder, *op, left, right)
            },
            JitProgram::Boolean { op, left, right } => self.logical(builder, *op, left, right),

            // Functions, in `functions`.
            JitProgram::Not(input) => self.not(builder, input),

            // Ternary, `when(...).then(...).otherwise(...)`, in `conditionals`.
            JitProgram::Ternary {
                predicate,
                truthy,
                falsy,
                dtype,
            } => self.ternary(builder, predicate, truthy, falsy, *dtype),
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
