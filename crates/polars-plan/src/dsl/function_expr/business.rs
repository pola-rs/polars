use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
use polars_ops::prelude::Roll;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::FunctionOptions;
use crate::dsl::{FieldsMapper, SpecialEq};
use crate::map_as_slice;
use crate::prelude::{ColumnsUdf, FunctionFlags};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum BusinessFunction {
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

impl BusinessFunction {
    pub fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        match self {
            Self::BusinessDayCount { .. } => mapper.with_dtype(DataType::Int32),
            Self::AddBusinessDay { .. } => mapper.with_same_dtype(),
            Self::IsBusinessDay { .. } => mapper.with_dtype(DataType::Boolean),
        }
    }
    pub fn function_options(&self) -> FunctionOptions {
        use BusinessFunction as B;
        match self {
            B::BusinessDayCount { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            B::AddBusinessDay { .. } | B::IsBusinessDay { .. } => FunctionOptions::elementwise(),
        }
    }
}

impl Display for BusinessFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use BusinessFunction::*;
        let s = match self {
            BusinessDayCount { .. } => "business_day_count",
            AddBusinessDay { .. } => "add_business_days",
            IsBusinessDay { .. } => "is_business_day",
        };
        write!(f, "{s}")
    }
}
impl From<BusinessFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: BusinessFunction) -> Self {
        use BusinessFunction::*;
        match func {
            BusinessDayCount {
                week_mask,
                holidays,
            } => {
                map_as_slice!(business_day_count, week_mask, &holidays)
            },
            AddBusinessDay {
                week_mask,
                holidays,
                roll,
            } => {
                map_as_slice!(add_business_days, week_mask, &holidays, roll)
            },
            IsBusinessDay {
                week_mask,
                holidays,
            } => {
                map_as_slice!(is_business_day, week_mask, &holidays)
            },
        }
    }
}

pub(super) fn business_day_count(
    s: &[Column],
    week_mask: [bool; 7],
    holidays: &[i32],
) -> PolarsResult<Column> {
    let start = &s[0];
    let end = &s[1];
    polars_ops::prelude::business_day_count(
        start.as_materialized_series(),
        end.as_materialized_series(),
        week_mask,
        holidays,
    )
    .map(Column::from)
}
pub(super) fn add_business_days(
    s: &[Column],
    week_mask: [bool; 7],
    holidays: &[i32],
    roll: Roll,
) -> PolarsResult<Column> {
    let start = &s[0];
    let n = &s[1];
    polars_ops::prelude::add_business_days(
        start.as_materialized_series(),
        n.as_materialized_series(),
        week_mask,
        holidays,
        roll,
    )
    .map(Column::from)
}

pub(super) fn is_business_day(
    s: &[Column],
    week_mask: [bool; 7],
    holidays: &[i32],
) -> PolarsResult<Column> {
    let dates = &s[0];
    polars_ops::prelude::is_business_day(dates.as_materialized_series(), week_mask, holidays)
        .map(Column::from)
}
