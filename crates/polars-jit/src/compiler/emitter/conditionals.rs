//! `when(...).then(...).otherwise(...)`.

use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::values::{JitValue, append_value_params, value_args, value_from_params};
use crate::jit_program::{JitProgram, ScalarType};

impl Emitter<'_> {
    /// Evaluate only the selected branch; a null predicate selects `falsy`, like Polars.
    pub(super) fn ternary(
        &mut self,
        builder: &mut FunctionBuilder<'_>,
        predicate: &JitProgram,
        truthy: &JitProgram,
        falsy: &JitProgram,
        dtype: ScalarType,
    ) -> PolarsResult<JitValue> {
        let truthy_block = builder.create_block();
        let falsy_block = builder.create_block();
        let merge = builder.create_block();
        append_value_params(builder, merge, dtype);
        let predicate = self.emit(builder, predicate)?;
        let take_truthy = builder.ins().band(predicate.data.scalar(), predicate.valid);
        builder
            .ins()
            .brif(take_truthy, truthy_block, &[], falsy_block, &[]);
        builder.switch_to_block(truthy_block);
        let truthy = self.emit(builder, truthy)?;
        builder.ins().jump(merge, &value_args(truthy));
        builder.switch_to_block(falsy_block);
        let falsy = self.emit(builder, falsy)?;
        builder.ins().jump(merge, &value_args(falsy));
        builder.switch_to_block(merge);
        Ok(value_from_params(builder.block_params(merge), dtype))
    }
}
