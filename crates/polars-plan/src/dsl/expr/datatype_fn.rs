use std::fmt;

use crate::dsl::{DataTypeExpr, DataTypeSelector};

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DataTypeFunction {
    /// Get a `str` repr of the DataType expression.
    Display(DataTypeExpr),
    /// Return whether two datatype expressions are equal.
    Eq(DataTypeExpr, DataTypeExpr),

    /// Return a boolean literal signifying whether the datatype is a specific kind.
    Matches(DataTypeExpr, DataTypeSelector),

    Array(DataTypeExpr, ArrayDataTypeFunction),
    Struct(DataTypeExpr, StructDataTypeFunction),
}

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ArrayDataTypeFunction {
    Width,
    /// Get a list literal with the dimensions of array nestings.
    Shape,
}

#[derive(PartialEq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum StructDataTypeFunction {
    FieldNames,
}

impl fmt::Debug for DataTypeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Display(dt_expr) => write!(f, "{dt_expr:?}.display()"),
            Self::Eq(l, r) => write!(f, "[{l:?} == {r:?}]"),
            Self::Matches(dt_expr, selector) => {
                write!(f, "{dt_expr:?}.matches({selector})")
            },
            Self::Array(dt_expr, t) => {
                fmt::Debug::fmt(dt_expr, f)?;
                f.write_str(match t {
                    ArrayDataTypeFunction::Width => ".arr.width()",
                    ArrayDataTypeFunction::Shape => ".arr.shape()",
                })
            },
            Self::Struct(dt_expr, t) => {
                fmt::Debug::fmt(dt_expr, f)?;
                match t {
                    StructDataTypeFunction::FieldNames => f.write_str(".struct.fields()"),
                }
            },
        }
    }
}
