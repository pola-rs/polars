//! `when(...).then(...).otherwise(...)`.

use cranelift_codegen::ir::{InstBuilder, types};
use cranelift_frontend::FunctionBuilder;
use polars_error::PolarsResult;

use super::Emitter;
use crate::compiler::helpers::cranelift_type;
use crate::compiler::values::JitValue;
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
        builder.append_block_param(merge, cranelift_type(dtype));
        builder.append_block_param(merge, types::I8);
        let predicate = self.emit(builder, predicate)?;
        let take_truthy = builder.ins().band(predicate.data, predicate.valid);
        builder
            .ins()
            .brif(take_truthy, truthy_block, &[], falsy_block, &[]);
        builder.switch_to_block(truthy_block);
        let truthy = self.emit(builder, truthy)?;
        builder
            .ins()
            .jump(merge, &[truthy.data.into(), truthy.valid.into()]);
        builder.switch_to_block(falsy_block);
        let falsy = self.emit(builder, falsy)?;
        builder
            .ins()
            .jump(merge, &[falsy.data.into(), falsy.valid.into()]);
        builder.switch_to_block(merge);
        Ok(JitValue {
            data: builder.block_params(merge)[0],
            valid: builder.block_params(merge)[1],
        })
    }
}
