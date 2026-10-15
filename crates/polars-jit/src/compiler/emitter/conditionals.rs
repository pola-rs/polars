//! `when(...).then(...).otherwise(...)`.

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::helpers::cranelift_type;
use crate::jit_program::{JitProgram, ScalarType};

impl Emitter<'_> {
    /// Evaluate only the selected branch and merge its value.
    pub(super) fn ternary(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        predicate: &JitProgram,
        truthy: &JitProgram,
        falsy: &JitProgram,
        dtype: ScalarType,
    ) -> PolarsResult<Value> {
        let truthy_block = builder.create_block();
        let falsy_block = builder.create_block();
        let merge = builder.create_block();
        builder.append_block_param(merge, cranelift_type(dtype));
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
