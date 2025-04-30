use std::fmt;
use std::sync::Arc;

use polars_core::prelude::*;
use strum_macros::IntoStaticStr;

use super::{ColumnsUdf, SpecialEq};
use crate::dsl::{FieldsMapper, FunctionOptions};
use crate::map;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum BitwiseFunction {
    CountOnes,
    CountZeros,

    LeadingOnes,
    LeadingZeros,

    TrailingOnes,
    TrailingZeros,

    // Bitwise Aggregations
    And,
    Or,
    Xor,
}

impl fmt::Display for BitwiseFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        use BitwiseFunction as B;

        let s = match self {
            B::CountOnes => "count_ones",
            B::CountZeros => "count_zeros",
            B::LeadingOnes => "leading_ones",
            B::LeadingZeros => "leading_zeros",
            B::TrailingOnes => "trailing_ones",
            B::TrailingZeros => "trailing_zeros",

            B::And => "and",
            B::Or => "or",
            B::Xor => "xor",
        };

        f.write_str(s)
    }
}

impl From<BitwiseFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: BitwiseFunction) -> Self {
        use BitwiseFunction as B;

        match func {
            B::CountOnes => map!(count_ones),
            B::CountZeros => map!(count_zeros),
            B::LeadingOnes => map!(leading_ones),
            B::LeadingZeros => map!(leading_zeros),
            B::TrailingOnes => map!(trailing_ones),
            B::TrailingZeros => map!(trailing_zeros),

            B::And => map!(reduce_and),
            B::Or => map!(reduce_or),
            B::Xor => map!(reduce_xor),
        }
    }
}

impl BitwiseFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        mapper.try_map_dtype(|dtype| {
            let is_valid = match dtype {
                DataType::Boolean => true,
                dt if dt.is_integer() => true,
                dt if dt.is_float() => true,
                _ => false,
            };

            if !is_valid {
                polars_bail!(InvalidOperation: "dtype {} not supported in '{}' operation", dtype, self);
            }

            match self {
                Self::CountOnes |
                Self::CountZeros |
                Self::LeadingOnes |
                Self::LeadingZeros |
                Self::TrailingOnes |
                Self::TrailingZeros => Ok(DataType::UInt32),
                Self::And |
                Self::Or |
                Self::Xor => Ok(dtype.clone()),
            }
        })
    }

    pub fn function_options(&self) -> FunctionOptions {
        use BitwiseFunction as B;
        match self {
            B::CountOnes
            | B::CountZeros
            | B::LeadingOnes
            | B::LeadingZeros
            | B::TrailingOnes
            | B::TrailingZeros => FunctionOptions::elementwise(),
            B::And | B::Or | B::Xor => FunctionOptions::aggregation(),
        }
    }
}

fn count_ones(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::count_ones)
}

fn count_zeros(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::count_zeros)
}

fn leading_ones(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::leading_ones)
}

fn leading_zeros(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::leading_zeros)
}

fn trailing_ones(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::trailing_ones)
}

fn trailing_zeros(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::trailing_zeros)
}

fn reduce_and(c: &Column) -> PolarsResult<Column> {
    c.and_reduce().map(|v| v.into_column(c.name().clone()))
}

fn reduce_or(c: &Column) -> PolarsResult<Column> {
    c.or_reduce().map(|v| v.into_column(c.name().clone()))
}

fn reduce_xor(c: &Column) -> PolarsResult<Column> {
    c.xor_reduce().map(|v| v.into_column(c.name().clone()))
}
