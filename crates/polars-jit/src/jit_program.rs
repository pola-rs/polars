use polars_core::prelude::DataType;
use polars_defs::expr::RoundMode;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::pl_str::PlSmallStr;

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
    /// UTF-8 text, held as a pointer and a byte length.
    String,
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
            DataType::String => Self::String,
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
            Self::String => DataType::String,
        }
    }

    pub(crate) fn bit_width(self) -> u16 {
        match self {
            Self::Boolean => 8,
            Self::Int32 | Self::UInt32 | Self::Float32 => 32,
            Self::Int64 | Self::UInt64 | Self::Float64 => 64,
            // A pointer and a byte length on the 64-bit hosts the JIT supports.
            Self::String => 128,
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
        !matches!(self, Self::Boolean | Self::String)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    TrueDivide,
    FloorDivide,
    Modulus,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum JitLiteral {
    Boolean(bool),
    Signed(i128),
    Unsigned(u128),
    Float32(u32),
    Float64(u64),
    String(PlSmallStr),
}

/// A classification of a floating-point value. Integers are never NaN or infinite.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FloatCheck {
    /// `is_nan`; `is_not_nan` is its negation.
    Nan,
    /// `is_finite`: neither NaN nor infinite.
    Finite,
    /// `is_infinite`: positive or negative infinity.
    Infinite,
}

/// What a cast does with a value that does not fit its target type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CastOverflow {
    /// The cast cannot fail, or wraps like `wrap_numerical=True`.
    Unchecked,
    /// The result is null, like `strict=False`.
    Null,
    /// The fold stops with an error, like `strict=True`.
    Raise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompareOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    EqualMissing,
    NotEqualMissing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BooleanOp {
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum NumericUnary {
    Negate,
    Abs,
    Sign,
    Floor,
    Ceil,
    Round { decimals: u32, mode: RoundMode },
    Sqrt,
}

/// A scalar calculation independent of how its inputs are supplied.
#[derive(Clone, Debug, PartialEq)]
pub enum JitProgram {
    NumericUnary {
        op: NumericUnary,
        input: Box<Self>,
        dtype: ScalarType,
    },
    Pow {
        base: Box<Self>,
        exponent: Box<Self>,
        dtype: ScalarType,
    },
    Clip {
        input: Box<Self>,
        min: Option<Box<Self>>,
        max: Option<Box<Self>>,
        dtype: ScalarType,
    },
    MinMax {
        left: Box<Self>,
        right: Box<Self>,
        max: bool,
        dtype: ScalarType,
    },
    Input {
        input: InputId,
        dtype: ScalarType,
    },
    Literal {
        value: JitLiteral,
        dtype: ScalarType,
    },
    Null {
        dtype: ScalarType,
    },
    Cast {
        input: Box<Self>,
        dtype: ScalarType,
        overflow: CastOverflow,
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
    IsNull(Box<Self>),
    FloatCheck {
        check: FloatCheck,
        input: Box<Self>,
    },
    /// `value`, or `fill` where `value` is null. Both already have type `dtype`.
    FillNull {
        value: Box<Self>,
        fill: Box<Self>,
        dtype: ScalarType,
    },
    /// `when(predicate).then(truthy).otherwise(falsy)`, Polars' `Expr::Ternary`. Only the
    /// selected branch is evaluated.
    Ternary {
        predicate: Box<Self>,
        truthy: Box<Self>,
        falsy: Box<Self>,
        dtype: ScalarType,
    },
    /// The length of a string as UInt32: in bytes, or in Unicode scalar values (`chars`).
    StringLength {
        input: Box<Self>,
        chars: bool,
    },
}

impl JitProgram {
    pub(crate) fn dtype(&self) -> ScalarType {
        match self {
            Self::Input { dtype, .. }
            | Self::NumericUnary { dtype, .. }
            | Self::Pow { dtype, .. }
            | Self::Clip { dtype, .. }
            | Self::MinMax { dtype, .. }
            | Self::Literal { dtype, .. }
            | Self::Null { dtype }
            | Self::Cast { dtype, .. }
            | Self::Binary { dtype, .. }
            | Self::Ternary { dtype, .. }
            | Self::FillNull { dtype, .. } => *dtype,
            Self::StringLength { .. } => ScalarType::UInt32,
            Self::Compare { .. }
            | Self::Boolean { .. }
            | Self::Not(_)
            | Self::IsNull(_)
            | Self::FloatCheck { .. } => ScalarType::Boolean,
        }
    }
}
