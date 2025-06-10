use std::fmt;

use polars_ops::prelude::Roll;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
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

impl fmt::Display for BusinessFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        use BusinessFunction::*;
        let s = match self {
            BusinessDayCount { .. } => "business_day_count",
            AddBusinessDay { .. } => "add_business_days",
            IsBusinessDay { .. } => "is_business_day",
        };
        write!(f, "{s}")
    }
}
