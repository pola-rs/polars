use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
use polars_defs::expr::Roll;

use super::FunctionOptions;
use crate::plans::aexpr::function_expr::FieldsMapper;
use crate::prelude::FunctionFlags;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRBusinessFunction {
    BusinessDayCount { week_mask: [bool; 7] },
    AddBusinessDay { week_mask: [bool; 7], roll: Roll },
    IsBusinessDay { week_mask: [bool; 7] },
}

impl<'a> FieldsMapper<'a> {
    /// Validate that the first `n` dtypes are Date, or Date/Datetime if `allow_datetime`.
    fn ensure_is_business_date(&self, n: usize, allow_datetime: bool) -> PolarsResult<()> {
        for field in &self.args()[..n] {
            let dt = field.dtype();
            polars_ensure!(
                matches!(dt, DataType::Date | DataType::Unknown(UnknownKind::Any))
                    || (allow_datetime && matches!(dt, DataType::Datetime(_, _))),
                InvalidOperation:
                "expected {}, got {dt}",
                if allow_datetime { "date or datetime" } else { "date" }
            );
        }
        Ok(())
    }
}

impl IRBusinessFunction {
    pub fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        match self {
            Self::BusinessDayCount { .. } => mapper.ensure_is_business_date(2, false)?,
            Self::AddBusinessDay { .. } | Self::IsBusinessDay { .. } => {
                mapper.ensure_is_business_date(1, true)?
            },
        }
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
