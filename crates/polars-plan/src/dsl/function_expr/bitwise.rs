use std::fmt;
use std::sync::Arc;

use polars_core::prelude::*;

use super::{ColumnsUdf, SpecialEq};
use crate::dsl::FieldsMapper;
use crate::map;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum BitwiseFunction {
    CountOnes,
    CountZeros,

    LeadingOnes,
    LeadingZeros,

    TrailingOnes,
    TrailingZeros,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum BitwiseAggFunction {
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
        }
    }
}

impl From<BitwiseAggFunction> for GroupByBitwiseMethod {
    fn from(value: BitwiseAggFunction) -> Self {
        match value {
            BitwiseAggFunction::And => Self::And,
            BitwiseAggFunction::Or => Self::Or,
            BitwiseAggFunction::Xor => Self::Xor,
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

            Ok(DataType::UInt32)
        })
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
