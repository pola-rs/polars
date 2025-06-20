use std::fmt;

use polars_utils::pl_str::PlSmallStr;

use crate::dsl::DataTypeExpr;

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DataTypeKind {
    /// Integer | Float | Decimal
    Numeric,
    /// SignedInteger | UnsignedInteger
    Integer,
    /// IntXX
    SignedInteger,
    /// UIntXX
    UnsignedInteger,
    Float,

    Decimal,

    Categorical,
    Enum,

    /// Array | List | Struct
    Nested,
    Array(Option<usize>),
    List,
    Struct,

    /// Date | Datetime | Duration | Time
    Temporal,
    Datetime,
    Duration,

    Object,
}

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DataTypeFunction {
    /// Get a `str` repr of the DataType expression.
    ToString(DataTypeExpr),
    /// Return whether two datatype expressions are equal.
    Eq(DataTypeExpr, DataTypeExpr),

    /// Return a boolean literal signifying whether the datatype is a specific kind.
    IsKind(DataTypeExpr, DataTypeKind),

    ElementBitSize(DataTypeExpr),

    Array(DataTypeExpr, ArrayDataTypeFunction),
    Struct(DataTypeExpr, StructDataTypeFunction),
    Enum(DataTypeExpr, EnumDataTypeFunction),
}

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ArrayDataTypeFunction {
    Width,
    /// Get a list literal with the dimensions of array nestings.
    Dimensions,
}

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum StructDataTypeFunction {
    NumFields,
    FieldNames,
    FieldName {
        idx: i64,
        raise_on_oob: bool,
    },
    FieldIndex {
        name: PlSmallStr,
        raise_on_missing: bool,
    },
}

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum EnumDataTypeFunction {
    NumCategories,
    Categories,
    GetCategory {
        idx: i64,
        raise_on_oob: bool,
    },
    IndexOfCategory {
        cat: PlSmallStr,
        raise_on_missing: bool,
    },
}

impl fmt::Debug for DataTypeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToString(dt_expr) => write!(f, "{dt_expr:?}.to_string()"),
            Self::Eq(l, r) => write!(f, "[{l:?} == {r:?}]"),
            Self::IsKind(dt_expr, kind) => {
                fmt::Debug::fmt(dt_expr, f)?;
                match kind {
                    DataTypeKind::Numeric => f.write_str(".is_numeric()"),
                    DataTypeKind::Integer => f.write_str(".is_integer()"),
                    DataTypeKind::SignedInteger => f.write_str(".int.is_signed()"),
                    DataTypeKind::UnsignedInteger => f.write_str(".int.is_unsigned()"),
                    DataTypeKind::Float => f.write_str(".is_float()"),
                    DataTypeKind::Decimal => f.write_str(".is_decimal()"),
                    DataTypeKind::Categorical => f.write_str(".is_categorical()"),
                    DataTypeKind::Enum => f.write_str(".is_enum()"),
                    DataTypeKind::Nested => f.write_str(".is_nested()"),
                    DataTypeKind::Array(None) => f.write_str(".is_array()"),
                    DataTypeKind::Array(Some(width)) => write!(f, ".arr.has_width({width})"),
                    DataTypeKind::List => f.write_str(".is_list()"),
                    DataTypeKind::Struct => f.write_str(".is_struct()"),
                    DataTypeKind::Temporal => f.write_str(".is_temporal()"),
                    DataTypeKind::Datetime => f.write_str(".is_datetime()"),
                    DataTypeKind::Duration => f.write_str(".is_duration()"),
                    DataTypeKind::Object => f.write_str(".is_object()"),
                }
            },
            Self::ElementBitSize(dt_expr) => write!(f, "{dt_expr:?}.element_bit_size()"),
            Self::Array(dt_expr, t) => {
                fmt::Debug::fmt(dt_expr, f)?;
                f.write_str(match t {
                    ArrayDataTypeFunction::Width => ".arr.width()",
                    ArrayDataTypeFunction::Dimensions => ".arr.dimensions()",
                })
            },
            Self::Struct(dt_expr, t) => {
                fmt::Debug::fmt(dt_expr, f)?;
                match t {
                    StructDataTypeFunction::NumFields => f.write_str(".struct.num_fields()"),
                    StructDataTypeFunction::FieldNames => f.write_str(".struct.fields()"),
                    StructDataTypeFunction::FieldName { idx, raise_on_oob } => {
                        write!(f, ".struct.field_name({idx}, raise_on_oob={raise_on_oob})")
                    },
                    StructDataTypeFunction::FieldIndex {
                        name,
                        raise_on_missing,
                    } => write!(
                        f,
                        ".struct.field_index({name}, raise_on_missing={raise_on_missing})"
                    ),
                }
            },
            Self::Enum(dt_expr, t) => {
                fmt::Debug::fmt(dt_expr, f)?;
                match t {
                    EnumDataTypeFunction::NumCategories => f.write_str(".enum.num_categories()"),
                    EnumDataTypeFunction::Categories => f.write_str(".enum.categories()"),
                    EnumDataTypeFunction::GetCategory { idx, raise_on_oob } => {
                        write!(f, ".enum.get_category({idx}, raise_on_oob={raise_on_oob})")
                    },
                    EnumDataTypeFunction::IndexOfCategory {
                        cat,
                        raise_on_missing,
                    } => write!(
                        f,
                        ".enum.index_of_category({cat}, raise_on_missing={raise_on_missing})"
                    ),
                }
            },
        }
    }
}
