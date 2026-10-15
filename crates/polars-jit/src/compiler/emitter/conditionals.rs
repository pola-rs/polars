//! `when(...).then(...).otherwise(...)`.

use cranelift_codegen::ir::{InstBuilder, Value, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::jit_program::JitProgram;

impl Emitter<'_> {
    /// Evaluate only the selected branch and merge its value.
    pub(super) fn ternary(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        predicate: &JitProgram,
        truthy: &JitProgram,
        falsy: &JitProgram,
    ) -> PolarsResult<Value> {
        let truthy_block = builder.create_block();
        let falsy_block = builder.create_block();
        let merge = builder.create_block();
        builder.append_block_param(merge, types::I64);
        let predicate = self.emit(builder, predicate)?;
        builder
            .ins()
            .brif(predicate, truthy_block, &[], falsy_block, &[]);
        builder.switch_to_block(truthy_block);
        let truthy = self.emit(builder, truthy)?;
        builder.ins().jump(merge, &[truthy.into()]);
        builder.switch_to_block(falsy_block);
        let falsy = self.emit(builder, falsy)?;
        builder.ins().jump(merge, &[falsy.into()]);
        builder.switch_to_block(merge);
        Ok(builder.block_params(merge)[0])
    }
}
