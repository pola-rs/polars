//! The generated function: one loop over the rows that evaluates the fold body for each row.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use polars_error::{PolarsResult, polars_ensure, polars_err};

use super::emitter::Emitter;
use super::helpers::{compile_error, cranelift_type};
use super::types::{ListFunction, OwnedModule};
use super::{CompiledProgram, ListBindings};
use crate::jit_program::JitProgram;

pub(super) fn compile(
    program: &JitProgram,
    bindings: &ListBindings,
) -> PolarsResult<CompiledProgram> {
    polars_ensure!(bindings.row_inputs.len() == 1, InvalidOperation:
        "list compilation currently requires one row input");
    let state_type = program.input_dtype(bindings.state_input).ok_or_else(
        || polars_err!(InvalidOperation: "foldv program does not reference its state"),
    )?;
    let row_type = program.input_dtype(bindings.row_inputs[0]).ok_or_else(
        || polars_err!(InvalidOperation: "foldv program does not reference its row input"),
    )?;
    polars_ensure!(program.dtype() == state_type, InvalidOperation:
        "foldv program output type must match its state type");

    let mut owner = host_module()?;
    let module = owner.module_mut();
    let pointer_type = module.target_config().pointer_type();

    let mut context = module.make_context();
    context.func.signature = module.make_signature();
    context.func.signature.params = vec![AbiParam::new(pointer_type); 4];
    let id = module
        .declare_function("jit_list_program", Linkage::Local, &context.func.signature)
        .map_err(compile_error)?;

    let mut frontend = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
        let entry = builder.create_block();
        let loop_header = builder.create_block();
        let apply = builder.create_block();
        let done = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.append_block_param(loop_header, pointer_type);
        builder.append_block_param(loop_header, cranelift_type(state_type));

        builder.switch_to_block(entry);
        let params = builder.block_params(entry).to_vec();
        let zero = builder.ins().iconst(pointer_type, 0);
        let initial = builder.ins().load(
            cranelift_type(state_type),
            MemFlagsData::new(),
            params[2],
            0,
        );
        builder
            .ins()
            .jump(loop_header, &[zero.into(), initial.into()]);

        builder.switch_to_block(loop_header);
        let index = builder.block_params(loop_header)[0];
        let state = builder.block_params(loop_header)[1];
        let has_row = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, index, params[1]);
        builder.ins().brif(has_row, apply, &[], done, &[]);

        builder.switch_to_block(apply);
        let byte_offset = builder
            .ins()
            .imul_imm_u(index, row_type.byte_width() as i64);
        let values_pointer = builder
            .ins()
            .load(pointer_type, MemFlagsData::new(), params[0], 0);
        let address = builder.ins().iadd(values_pointer, byte_offset);
        let row_value =
            builder
                .ins()
                .load(cranelift_type(row_type), MemFlagsData::new(), address, 0);
        let values = [
            (bindings.state_input, state),
            (bindings.row_inputs[0], row_value),
        ];
        let next_state = Emitter::new(&values).emit(&mut builder, program)?;
        let output_offset = builder
            .ins()
            .imul_imm_u(index, state_type.byte_width() as i64);
        let output_address = builder.ins().iadd(params[3], output_offset);
        builder
            .ins()
            .store(MemFlagsData::new(), next_state, output_address, 0);
        let next_index = builder.ins().iadd_imm_u(index, 1);
        builder
            .ins()
            .jump(loop_header, &[next_index.into(), next_state.into()]);

        builder.switch_to_block(done);
        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize(module.target_config());
    }

    module
        .define_function(id, &mut context)
        .map_err(compile_error)?;
    module.finalize_definitions().map_err(compile_error)?;
    let address = module.get_finalized_function(id);
    // SAFETY: The generated ABI exactly matches ListFunction.
    let function = unsafe { std::mem::transmute::<*const u8, ListFunction>(address) };
    Ok(CompiledProgram {
        function,
        _module: owner,
    })
}

/// A JIT module for the CPU we are running on, optimizing for speed and verifying the generated
/// IR. It owns the executable memory, so it must outlive every call to the functions it
/// compiles.
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
    Ok(OwnedModule::new(JITModule::new(JITBuilder::with_isa(
        isa,
        cranelift_module::default_libcall_names(),
    ))))
}
