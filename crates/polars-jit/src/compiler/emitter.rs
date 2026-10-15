//! The `Emitter` generates Cranelift IR for one fold body inside the row loop.
//!
//! `emit` is the single place that matches `JitProgram` nodes: every arm calls the method that
//! generates that node. The methods live in the modules below, grouped by kind like the
//! translator's.

mod casts;
mod conditionals;
mod functions;
mod literals;
mod numbers;
mod operators;

use cranelift_codegen::ir::{Block, FuncRef, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use polars_error::{PolarsResult, polars_err};

use super::runtime_functions::RuntimeFunction;
use super::values::JitValue;
use crate::jit_program::{InputId, JitProgram, ScalarType};

/// What the compiled program must keep from emitting a fold body.
pub(super) struct Emitted {
    /// Source and target type of each strict cast, indexed like `CastFailure::cast`.
    pub(super) strict_casts: Box<[(ScalarType, ScalarType)]>,
    /// Bytes of string literals, which generated code reads through embedded pointers.
    pub(super) string_literals: Box<[Box<[u8]>]>,
}

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
    /// The module, for declaring runtime functions in the function on first use.
    module: &'a mut JITModule,
    /// Each runtime function's declaration in the module, indexed like `RuntimeFunction::ALL`.
    runtime_functions: &'a [FuncId],
    /// Each runtime function's declaration in the function, once used.
    function_refs: [Option<FuncRef>; RuntimeFunction::ALL.len()],
    /// Pointer to the call's `StringContext`.
    strings: Value,
    /// Bytes of string literals, kept alive by the compiled program.
    string_literals: Vec<Box<[u8]>>,
    /// Whether the body creates strings, which must be freed when each row ends.
    allocates_strings: bool,
}

impl<'a> Emitter<'a> {
    pub(super) fn new(
        inputs: &'a [(InputId, JitValue)],
        index: Value,
        cast_failure: Value,
        failure: Block,
        module: &'a mut JITModule,
        runtime_functions: &'a [FuncId],
        strings: Value,
    ) -> Self {
        Self {
            inputs,
            index,
            cast_failure,
            failure,
            strict_casts: Vec::new(),
            module,
            runtime_functions,
            function_refs: [None; RuntimeFunction::ALL.len()],
            strings,
            string_literals: Vec::new(),
            allocates_strings: false,
        }
    }

    /// Generate `program` at the builder's current position and return its value.
    pub(super) fn emit(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        program: &JitProgram,
    ) -> PolarsResult<JitValue> {
        match program {
            JitProgram::NumericUnary { op, input, dtype } => {
                self.numeric_unary(builder, *op, input, *dtype)
            },
            JitProgram::Pow {
                base,
                exponent,
                dtype,
            } => self.pow(builder, base, exponent, *dtype),
            JitProgram::Clip {
                input,
                min,
                max,
                dtype,
            } => self.clip(builder, input, min.as_deref(), max.as_deref(), *dtype),
            JitProgram::MinMax {
                left,
                right,
                max,
                dtype,
            } => self.min_max(builder, left, right, *max, *dtype),
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
                left,
                right,
                dtype: ScalarType::String,
                ..
            } => self.concat(builder, left, right),
            JitProgram::Binary {
                op,
                left,
                right,
                dtype,
            } => self.arithmetic(builder, *op, left, right, *dtype),
            JitProgram::Compare { op, left, right } if left.dtype() == ScalarType::String => {
                self.compare_strings(builder, *op, left, right)
            },
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
            JitProgram::StringLength { input, chars } => self.string_length(builder, input, *chars),
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

    /// Call a runtime function and return its result, if it has one.
    pub(super) fn call(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        function: RuntimeFunction,
        arguments: &[Value],
    ) -> Option<Value> {
        let reference = match self.function_refs[function as usize] {
            Some(reference) => reference,
            None => {
                let reference = self
                    .module
                    .declare_func_in_func(self.runtime_functions[function as usize], builder.func);
                self.function_refs[function as usize] = Some(reference);
                reference
            },
        };
        let call = builder.ins().call(reference, arguments);
        builder.inst_results(call).first().copied()
    }

    /// Whether the body creates strings, which must be freed when each row ends.
    pub(super) fn allocates_strings(&self) -> bool {
        self.allocates_strings
    }

    /// What the compiled program must keep once the body is emitted.
    pub(super) fn finish(self) -> Emitted {
        Emitted {
            strict_casts: self.strict_casts.into_boxed_slice(),
            string_literals: self.string_literals.into_boxed_slice(),
        }
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
