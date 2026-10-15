use polars_core::prelude::DataType;
use polars_error::{PolarsResult, polars_bail};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct InputId(pub(crate) u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScalarType {
    Boolean,
    Int32,
    Int64,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl ScalarType {
    pub(crate) fn from_dtype(dtype: &DataType) -> PolarsResult<Self> {
        Ok(match dtype {
            DataType::Boolean => Self::Boolean,
            DataType::Int32 => Self::Int32,
            DataType::Int64 => Self::Int64,
            DataType::UInt32 => Self::UInt32,
            DataType::UInt64 => Self::UInt64,
            DataType::Float32 => Self::Float32,
            DataType::Float64 => Self::Float64,
            _ => polars_bail!(InvalidOperation: "foldv does not support dtype {dtype}"),
        })
    }

    pub(crate) fn dtype(self) -> DataType {
        match self {
            Self::Boolean => DataType::Boolean,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::UInt32 => DataType::UInt32,
            Self::UInt64 => DataType::UInt64,
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
        }
    }

    pub(crate) fn bit_width(self) -> u16 {
        match self {
            Self::Boolean => 8,
            Self::Int32 | Self::UInt32 | Self::Float32 => 32,
            Self::Int64 | Self::UInt64 | Self::Float64 => 64,
        }
    }

    pub(crate) fn byte_width(self) -> usize {
        usize::from(self.bit_width() / 8)
    }

    pub(crate) fn is_signed(self) -> bool {
        matches!(self, Self::Int32 | Self::Int64)
    }

    pub(crate) fn is_float(self) -> bool {
        matches!(self, Self::Float32 | Self::Float64)
    }

    pub(crate) fn is_numeric(self) -> bool {
        !matches!(self, Self::Boolean)
    }
}

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
    Unsigned(u128),
    Float32(u32),
    Float64(u64),
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

/// A scalar calculation independent of how its inputs are supplied.
#[derive(Clone, Debug, PartialEq)]
pub enum JitProgram {
    Input {
        input: InputId,
        dtype: ScalarType,
    },
    Literal {
        value: JitLiteral,
        dtype: ScalarType,
    },
    Cast {
        input: Box<Self>,
        dtype: ScalarType,
    },
    Binary {
        op: BinaryOp,
        left: Box<Self>,
        right: Box<Self>,
        dtype: ScalarType,
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
        dtype: ScalarType,
    },
}

impl JitProgram {
    pub(crate) fn dtype(&self) -> ScalarType {
        match self {
            Self::Input { dtype, .. }
            | Self::Literal { dtype, .. }
            | Self::Cast { dtype, .. }
            | Self::Binary { dtype, .. }
            | Self::Ternary { dtype, .. } => *dtype,
            Self::Compare { .. } | Self::Boolean { .. } | Self::Not(_) => ScalarType::Boolean,
        }
    }
}
