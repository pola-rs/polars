#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct InputId(pub(crate) u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BinaryOp {
    Add,
}

use polars_core::prelude::DataType;

/// A scalar calculation independent of how its inputs are supplied.
#[derive(Clone, Debug, PartialEq)]
pub enum JitProgram {
    Input {
        input: InputId,
        dtype: DataType,
    },
    Binary {
        op: BinaryOp,
        left: Box<Self>,
        right: Box<Self>,
        dtype: DataType,
    },
}

impl JitProgram {
    pub fn dtype(&self) -> &DataType {
        match self {
            Self::Input { dtype, .. } | Self::Binary { dtype, .. } => dtype,
        }
    }
}
