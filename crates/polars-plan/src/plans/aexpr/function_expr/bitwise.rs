use std::fmt;

use polars_core::prelude::*;
use strum_macros::IntoStaticStr;

use crate::plans::aexpr::function_expr::{FieldsMapper, FunctionOptions};
use crate::prelude::FunctionFlags;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum IRBitwiseFunction {
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

impl IRBitwiseFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        mapper.try_map_dtype(|dtype| {
            let is_valid = dtype.is_bool() || dtype.is_integer();
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
        use IRBitwiseFunction as B;
        match self {
            B::CountOnes
            | B::CountZeros
            | B::LeadingOnes
            | B::LeadingZeros
            | B::TrailingOnes
            | B::TrailingZeros => FunctionOptions::elementwise(),
            B::And | B::Or | B::Xor => FunctionOptions::aggregation()
                .with_flags(|f| f | FunctionFlags::NON_ORDER_OBSERVING),
        }
    }
}

impl fmt::Display for IRBitwiseFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        use IRBitwiseFunction as B;

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
