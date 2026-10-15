#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct InputId(pub(crate) u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BinaryOp {
    Add,
    Subtract,
    Multiply,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum JitLiteral {
    Boolean(bool),
    Signed(i128),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompareOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BooleanOp {
    And,
    Or,
}

use polars_core::prelude::DataType;

/// A scalar calculation independent of how its inputs are supplied.
#[derive(Clone, Debug, PartialEq)]
pub enum JitProgram {
    Input {
        input: InputId,
        dtype: DataType,
    },
    Literal {
        value: JitLiteral,
        dtype: DataType,
    },
    Binary {
        op: BinaryOp,
        left: Box<Self>,
        right: Box<Self>,
        dtype: DataType,
    },
    Compare {
        op: CompareOp,
        left: Box<Self>,
        right: Box<Self>,
    },
    Boolean {
        op: BooleanOp,
        left: Box<Self>,
        right: Box<Self>,
    },
    Not(Box<Self>),
    /// `when(predicate).then(truthy).otherwise(falsy)`, Polars' `Expr::Ternary`. Only the
    /// selected branch is evaluated.
    Ternary {
        predicate: Box<Self>,
        truthy: Box<Self>,
        falsy: Box<Self>,
        dtype: DataType,
    },
}

impl JitProgram {
    pub fn dtype(&self) -> &DataType {
        match self {
            Self::Input { dtype, .. }
            | Self::Literal { dtype, .. }
            | Self::Binary { dtype, .. }
            | Self::Ternary { dtype, .. } => dtype,
            Self::Compare { .. } | Self::Boolean { .. } | Self::Not(_) => &DataType::Boolean,
        }
    }
}
