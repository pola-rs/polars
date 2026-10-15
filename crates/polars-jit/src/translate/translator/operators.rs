//! Translation of binary operators.

use polars_core::prelude::DataType;
use polars_error::PolarsResult;
use polars_plan::dsl::Expr;

use super::Translator;
use crate::jit_program::{BinaryOp, InputId, JitProgram};
use crate::translate::helpers::unsupported;

impl Translator<'_> {
    /// `+`. Only `element + column` is supported so far: the state plus the row input.
    pub(super) fn arithmetic(&self, left: &Expr, right: &Expr) -> PolarsResult<JitProgram> {
        let adds_row_to_state = matches!(left, Expr::Element)
            && matches!(right, Expr::Column(name) if name == &self.row_names[0]);
        if !adds_row_to_state {
            return Err(unsupported());
        }
        Ok(JitProgram::Binary {
            op: BinaryOp::Add,
            left: Box::new(JitProgram::Input {
                input: InputId(0),
                dtype: DataType::Int64,
            }),
            right: Box::new(JitProgram::Input {
                input: InputId(1),
                dtype: DataType::Int64,
            }),
            dtype: DataType::Int64,
        })
    }
}
