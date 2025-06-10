use core::fmt;

use polars_core::error::{polars_err, PolarsResult};
use polars_core::prelude::DataType;
use polars_core::schema::Schema;
use polars_utils::arena::Arena;

use super::Expr;
use crate::plans::{to_expr_ir_no_selectors};

#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum DataTypeExpr {
    Literal(DataType),
    OfExpr(Box<Expr>),
}

#[recursive::recursive]
fn into_datatype_impl(dt_expr: DataTypeExpr, schema: &Schema) -> PolarsResult<DataType> {
    Ok(match dt_expr {
        DataTypeExpr::Literal(dt) => dt,
        DataTypeExpr::OfExpr(expr) => {
            let mut arena = Arena::new();
            let e = to_expr_ir_no_selectors(*expr, &mut arena, schema)?.ok_or_else(
                || polars_err!(InvalidOperation: "using a selector in a datatype expr"),
            )?;
            arena
                .get(e.node())
                .to_dtype(schema, Default::default(), &arena)?
        },
    })
}

impl DataTypeExpr {
    pub fn into_datatype(self, schema: &Schema) -> PolarsResult<DataType> {
        into_datatype_impl(self, schema)
    }

    pub fn as_literal(&self) -> Option<&DataType> {
        match self {
            Self::Literal(dt) => Some(dt),
            _ => None,
        }
    }
}

impl fmt::Debug for DataTypeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataTypeExpr::Literal(data_type) => data_type.fmt(f),
            DataTypeExpr::OfExpr(expr) => write!(f, "dtype_of({expr:?})"),
        }
    }
}

impl From<DataType> for DataTypeExpr {
    fn from(value: DataType) -> Self {
        Self::Literal(value)
    }
}
