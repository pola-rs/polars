//! The generated function: one loop over the rows that evaluates the fold body for each row.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlagsData, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use polars_error::{PolarsResult, polars_ensure, polars_err};

use super::emitter::Emitter;
use super::helpers::compile_error;
use super::runtime_functions::RuntimeFunction;
use super::types::{ListFunction, OwnedModule};
use super::values::{
    JitData, JitValue, append_value_params, load_words, value_args, value_from_params,
};
use super::{CompiledProgram, ListBindings};
use crate::jit_program::{JitProgram, ScalarType};

pub(super) fn compile(
    program: &JitProgram,
    bindings: &ListBindings,
) -> PolarsResult<CompiledProgram> {
    polars_ensure!(!bindings.rows.is_empty(), InvalidOperation:
        "list compilation requires at least one row input");
    let state_type = bindings.state.dtype;
    polars_ensure!(program.dtype() == state_type, InvalidOperation:
        "foldv program output type must match its state type");

    let mut owner = host_module()?;
    let module = owner.module_mut();
    let pointer_type = module.target_config().pointer_type();
    // Strings and the runtime functions use 64-bit words.
    polars_ensure!(pointer_type == types::I64, ComputeError: "the foldv JIT requires a 64-bit host");
    let runtime_functions = RuntimeFunction::declare_all(module)?;

    // Declare the native signature. It must match `ListFunction` parameter for parameter.
    // The first four parameters point to tables with one pointer-sized entry per row input.
    let mut context = module.make_context();
    context.func.signature = module.make_signature();
    context.func.signature.params = vec![
        AbiParam::new(pointer_type), // value_pointers: each input's typed value buffer.
        AbiParam::new(pointer_type), // value_bit_offsets: start bit of each Boolean value bitmap.
        AbiParam::new(pointer_type), // validity_pointers: each input's Arrow validity bitmap.
        AbiParam::new(pointer_type), // validity_bit_offsets: start bit of each validity bitmap.
        AbiParam::new(pointer_type), // row_count: number of rows shared by every input.
        AbiParam::new(pointer_type), // initial_state: pointer to the initial state value.
        AbiParam::new(types::I8),    // initial_valid: `1` if the initial state is valid, else `0`.
        AbiParam::new(pointer_type), // output: typed state value written for each row.
        AbiParam::new(pointer_type), // output_validity: one validity byte written for each row.
        AbiParam::new(pointer_type), // cast_failure: `CastFailure` written if a strict cast fails.
        AbiParam::new(pointer_type), // strings: the call's `StringContext`.
    ];
    // Returns the failing row if a strict cast fails, otherwise `row_count`.
    context.func.signature.returns = vec![AbiParam::new(pointer_type)];
    let id = module
        .declare_function("jit_list_program", Linkage::Local, &context.func.signature)
        .map_err(compile_error)?;

    // The generated function is one loop over the rows:
    //
    //   entry:       load the initial state, jump to loop_header(0, state, state_valid)
    //   loop_header: if index < row_count goto apply, else goto done
    //   apply:       load row values, evaluate the body, store the new state (the runtime keeps a
    //                string state), jump to loop_header(index + 1, new_state, new_state_valid)
    //   done:        return row_count
    //   failure:     return the failing row (reached only from a strict cast in the body)
    //
    // Cranelift has no mutable variables, so the row index and the carried state (with its
    // validity) are passed between iterations as parameters of `loop_header`.
    let mut frontend = FunctionBuilderContext::new();
    let emitted = {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
        let entry = builder.create_block();
        let loop_header = builder.create_block();
        let apply = builder.create_block();
        let done = builder.create_block();
        let failure = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.append_block_param(failure, pointer_type);
        builder.append_block_param(loop_header, pointer_type);
        append_value_params(&mut builder, loop_header, state_type);

        builder.switch_to_block(entry);
        // Parameter order matches `ListFunction`.
        let &[
            value_pointers,
            value_bit_offsets,
            validity_pointers,
            validity_bit_offsets,
            row_count,
            initial_state,
            initial_valid,
            output,
            output_validity,
            cast_failure,
            strings,
        ] = builder.block_params(entry)
        else {
            unreachable!("the JIT signature declares eleven parameters")
        };
        // Entry: read the initial state from memory and start the loop at row 0. Its validity
        // arrives as a plain byte argument rather than through memory.
        let zero = builder.ins().iconst(pointer_type, 0);
        let initial = JitValue {
            data: load_words(&mut builder, state_type, initial_state),
            valid: initial_valid,
        };
        let mut arguments = vec![BlockArg::from(zero)];
        arguments.extend(value_args(initial));
        builder.ins().jump(loop_header, &arguments);

        // Loop header: receive this iteration's row index and carried state, and exit once
        // every row has been processed. An empty input goes straight to `done`.
        builder.switch_to_block(loop_header);
        let index = builder.block_params(loop_header)[0];
        let state = value_from_params(&builder.block_params(loop_header)[1..], state_type);
        let has_row = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, index, row_count);
        builder.ins().brif(has_row, apply, &[], done, &[]);

        // Apply: collect every value the body can read, keyed by `InputId`. The carried state
        // comes first, followed by one entry per row input.
        builder.switch_to_block(apply);
        let mut values = Vec::with_capacity(bindings.rows.len() + 1);
        values.push((bindings.state.input, state));
        for (position, binding) in bindings.rows.iter().enumerate() {
            // Each per-input table holds one pointer-sized entry per row input, so this input's
            // buffer pointers and bit offsets are at `position * pointer size` in every table.
            // These loads are the same on every row but are currently repeated per iteration.
            let pointer_offset = i32::try_from(position * pointer_type.bytes() as usize)
                .map_err(|_| polars_err!(InvalidOperation: "too many JIT row inputs"))?;
            let values_pointer = builder.ins().load(
                pointer_type,
                MemFlagsData::new(),
                value_pointers,
                pointer_offset,
            );
            let value_bit_offset = builder.ins().load(
                pointer_type,
                MemFlagsData::new(),
                value_bit_offsets,
                pointer_offset,
            );
            let validity_pointer = builder.ins().load(
                pointer_type,
                MemFlagsData::new(),
                validity_pointers,
                pointer_offset,
            );
            let validity_bit_offset = builder.ins().load(
                pointer_type,
                MemFlagsData::new(),
                validity_bit_offsets,
                pointer_offset,
            );
            // Read this row's value. Arrow bit-packs Boolean values, so they are read from a
            // bitmap. Other types sit at `values + index * byte_width`; for strings the runtime
            // decodes Arrow's string views into `(pointer, length)` pairs first.
            let row_data = if binding.dtype == ScalarType::Boolean {
                JitData::Scalar(load_bitmap_bit(
                    &mut builder,
                    values_pointer,
                    value_bit_offset,
                    index,
                ))
            } else {
                let byte_offset = builder
                    .ins()
                    .imul_imm_u(index, binding.dtype.byte_width() as i64);
                let address = builder.ins().iadd(values_pointer, byte_offset);
                load_words(&mut builder, binding.dtype, address)
            };
            // Read this row's validity bit. Inputs without nulls point at an all-ones bitmap
            // supplied by the runtime, so this load never needs a branch.
            let row_valid =
                load_bitmap_bit(&mut builder, validity_pointer, validity_bit_offset, index);
            values.push((
                binding.input,
                JitValue {
                    data: row_data,
                    valid: row_valid,
                },
            ));
        }
        // Emit the fold body inline with the loop. Conditionals and strict casts inside it add
        // their own blocks, so the builder may now be in a later block than `apply`.
        let mut emitter = Emitter::new(
            &values,
            index,
            cast_failure,
            failure,
            &mut *module,
            &runtime_functions,
            strings,
        );
        let next_state = emitter.emit(&mut builder, program)?;

        let next_state = if state_type == ScalarType::String {
            // The runtime appends a string state to the output and keeps a copy for the next row,
            // since the new state may point into storage that the row's end frees.
            let (ptr, len) = next_state.data.string();
            let kept = emitter
                .call(
                    &mut builder,
                    RuntimeFunction::EndStringRow,
                    &[strings, ptr, len, next_state.valid],
                )
                .expect("end_string_row returns the kept state");
            let zero = builder.ins().iconst(types::I64, 0);
            let len = builder.ins().select(next_state.valid, len, zero);
            JitValue {
                data: JitData::Str { ptr: kept, len },
                valid: next_state.valid,
            }
        } else {
            // Record the new state as this row's output: its value in the typed output buffer and
            // its validity in the byte-per-row validity buffer, which the runtime turns into a
            // column.
            let output_offset = builder
                .ins()
                .imul_imm_u(index, state_type.byte_width() as i64);
            let output_address = builder.ins().iadd(output, output_offset);
            builder.ins().store(
                MemFlagsData::new(),
                next_state.data.scalar(),
                output_address,
                0,
            );
            let output_validity_address = builder.ins().iadd(output_validity, index);
            builder.ins().store(
                MemFlagsData::new(),
                next_state.valid,
                output_validity_address,
                0,
            );
            // Strings the body created are no longer needed once the row ends.
            if emitter.allocates_strings() {
                emitter.call(&mut builder, RuntimeFunction::EndRow, &[strings]);
            }
            next_state
        };
        let emitted = emitter.finish();
        // Carry the new state and its validity into the next iteration.
        let next_index = builder.ins().iadd_imm_u(index, 1);
        let mut arguments = vec![BlockArg::from(next_index)];
        arguments.extend(value_args(next_state));
        builder.ins().jump(loop_header, &arguments);

        // Done: every result has already been written to the output buffers, so report that
        // all rows were processed.
        builder.switch_to_block(done);
        builder.ins().return_(&[row_count]);

        // Failure: a strict cast has filled in `cast_failure`; report the row it failed on.
        // Sealing tells Cranelift that every block's predecessors are known so it can finish SSA
        // construction.
        builder.switch_to_block(failure);
        let failed_row = builder.block_params(failure)[0];
        builder.ins().return_(&[failed_row]);
        builder.seal_all_blocks();
        builder.finalize(module.target_config());
        emitted
    };

    // Compile the IR to machine code, make it executable, and turn its address into a callable
    // function pointer. The returned program keeps `owner` so the code stays mapped.
    module
        .define_function(id, &mut context)
        .map_err(compile_error)?;
    module.finalize_definitions().map_err(compile_error)?;
    let address = module.get_finalized_function(id);
    // SAFETY: The generated ABI exactly matches ListFunction.
    let function = unsafe { std::mem::transmute::<*const u8, ListFunction>(address) };
    Ok(CompiledProgram {
        function,
        state_type,
        input_types: bindings.rows.iter().map(|binding| binding.dtype).collect(),
        strict_casts: emitted.strict_casts,
        _string_literals: emitted.string_literals,
        _module: owner,
    })
}

/// A JIT module for the CPU we are running on, optimizing for speed and verifying the generated
/// IR, with every runtime function registered. It owns the executable memory, so it must
/// outlive every call to the functions it compiles.
fn host_module() -> PolarsResult<OwnedModule> {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").map_err(compile_error)?;
    flags
        .set("enable_verifier", "true")
        .map_err(compile_error)?;
    let isa = cranelift_native::builder()
        .map_err(compile_error)?
        .finish(settings::Flags::new(flags))
        .map_err(compile_error)?;
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    RuntimeFunction::register_all(&mut jit_builder);
    Ok(OwnedModule::new(JITModule::new(jit_builder)))
}

/// One bit of an Arrow bitmap, as `0` or `1`.
fn load_bitmap_bit(
    builder: &mut FunctionBuilder<'_>,
    bitmap: Value,
    bit_offset: Value,
    index: Value,
) -> Value {
    let bit_index = builder.ins().iadd(bit_offset, index);
    let byte_index = builder.ins().ushr_imm_u(bit_index, 3);
    let address = builder.ins().iadd(bitmap, byte_index);
    let byte = builder
        .ins()
        .load(types::I8, MemFlagsData::new(), address, 0);
    let shift = builder.ins().ireduce(types::I8, bit_index);
    let shifted = builder.ins().ushr(byte, shift);
    builder.ins().band_imm_u(shifted, 1)
}
