use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
use polars_ops::prelude::Roll;

use super::FunctionOptions;
use crate::plans::aexpr::function_expr::FieldsMapper;
use crate::prelude::FunctionFlags;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRBusinessFunction {
    BusinessDayCount {
        week_mask: [bool; 7],
        holidays: Vec<i32>,
    },
    AddBusinessDay {
        week_mask: [bool; 7],
        holidays: Vec<i32>,
        roll: Roll,
    },
    IsBusinessDay {
        week_mask: [bool; 7],
        holidays: Vec<i32>,
    },
}

impl IRBusinessFunction {
    pub fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        match self {
            Self::BusinessDayCount { .. } => mapper.with_dtype(DataType::Int32),
            Self::AddBusinessDay { .. } => mapper.with_same_dtype(),
            Self::IsBusinessDay { .. } => mapper.with_dtype(DataType::Boolean),
        }
    }
    pub fn function_options(&self) -> FunctionOptions {
        use IRBusinessFunction as B;
        match self {
            B::BusinessDayCount { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            B::AddBusinessDay { .. } | B::IsBusinessDay { .. } => FunctionOptions::elementwise(),
        }
    }
}

impl Display for IRBusinessFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRBusinessFunction::*;
        let s = match self {
            BusinessDayCount { .. } => "business_day_count",
            AddBusinessDay { .. } => "add_business_days",
            IsBusinessDay { .. } => "is_business_day",
        };
        write!(f, "{s}")
    }
}
