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

use cranelift_codegen::ir::{Block, Value};
use cranelift_frontend::FunctionBuilder;
use polars_error::{PolarsResult, polars_err};

use super::values::JitValue;
use crate::jit_program::{InputId, JitProgram, ScalarType};

/// State shared while emitting one fold body.
pub(super) struct Emitter<'a> {
    /// The value of every input the body can read in the current row.
    inputs: &'a [(InputId, JitValue)],
    /// The current row, reported when a strict cast fails.
    index: Value,
    /// Pointer to the `CastFailure` that a failing strict cast fills in.
    cast_failure: Value,
    /// The block that returns the failing row.
    failure: Block,
    /// Source and target type of every strict cast emitted so far.
    strict_casts: Vec<(ScalarType, ScalarType)>,
}

impl<'a> Emitter<'a> {
    pub(super) fn new(
        inputs: &'a [(InputId, JitValue)],
        index: Value,
        cast_failure: Value,
        failure: Block,
    ) -> Self {
        Self {
            inputs,
            index,
            cast_failure,
            failure,
            strict_casts: Vec::new(),
        }
    }

    /// Generate `program` at the builder's current position and return its value.
    pub(super) fn emit(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        program: &JitProgram,
    ) -> PolarsResult<JitValue> {
        match program {
            JitProgram::Input { input, .. } => self.input(*input),
            JitProgram::Literal { value, dtype } => self.literal(builder, value, *dtype),
            JitProgram::Null { dtype } => Ok(self.null(builder, *dtype)),
            JitProgram::Cast {
                input,
                dtype,
                overflow,
            } => self.cast(builder, input, *dtype, *overflow),

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
            JitProgram::IsNull(input) => self.is_null(builder, input),
            JitProgram::FloatCheck { check, input } => self.float_check(builder, *check, input),
            JitProgram::FillNull { value, fill, dtype } => {
                self.fill_null(builder, value, fill, *dtype)
            },

            // Ternary, `when(...).then(...).otherwise(...)`, in `conditionals`.
            JitProgram::Ternary {
                predicate,
                truthy,
                falsy,
                dtype,
            } => self.ternary(builder, predicate, truthy, falsy, *dtype),
        }
    }

    /// The strict casts the compiled program must keep, indexed like `CastFailure::cast`.
    pub(super) fn finish(self) -> Box<[(ScalarType, ScalarType)]> {
        self.strict_casts.into_boxed_slice()
    }

    /// The carried state or a row input, already loaded for the current row.
    fn input(&self, input: InputId) -> PolarsResult<JitValue> {
        self.inputs
            .iter()
            .find(|(candidate, _)| *candidate == input)
            .map(|(_, value)| *value)
            .ok_or_else(|| polars_err!(InvalidOperation: "missing JIT input {input:?}"))
    }
}
