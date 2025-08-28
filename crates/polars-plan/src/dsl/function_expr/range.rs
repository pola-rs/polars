use std::fmt;

use polars_core::prelude::*;
use polars_ops::series::ClosedInterval;
#[cfg(feature = "temporal")]
use polars_time::{ClosedWindow, Duration};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{DataTypeExpr, FunctionExpr};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum RangeFunction {
    IntRange {
        step: i64,
        dtype: DataTypeExpr,
    },
    IntRanges {
        dtype: DataTypeExpr,
    },
    LinearSpace {
        closed: ClosedInterval,
    },
    LinearSpaces {
        closed: ClosedInterval,
        array_width: Option<usize>,
    },
    #[cfg(feature = "dtype-date")]
    DateRange {
        interval: Duration,
        closed: ClosedWindow,
    },
    #[cfg(feature = "dtype-date")]
    DateRanges {
        interval: Duration,
        closed: ClosedWindow,
    },
    #[cfg(feature = "dtype-datetime")]
    DatetimeRange {
        interval: Duration,
        closed: ClosedWindow,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
    },
    #[cfg(feature = "dtype-datetime")]
    DatetimeRanges {
        interval: Duration,
        closed: ClosedWindow,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
    },
    #[cfg(feature = "dtype-time")]
    TimeRange {
        interval: Duration,
        closed: ClosedWindow,
    },
    #[cfg(feature = "dtype-time")]
    TimeRanges {
        interval: Duration,
        closed: ClosedWindow,
    },
}

impl From<RangeFunction> for FunctionExpr {
    fn from(value: RangeFunction) -> Self {
        Self::Range(value)
    }
}

impl fmt::Display for RangeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        use RangeFunction::*;
        let s = match self {
            IntRange { .. } => "int_range",
            IntRanges { .. } => "int_ranges",
            LinearSpace { .. } => "linear_space",
            LinearSpaces { .. } => "linear_spaces",
            #[cfg(feature = "dtype-date")]
            DateRange { .. } => "date_range",
            #[cfg(feature = "temporal")]
            DateRanges { .. } => "date_ranges",
            #[cfg(feature = "dtype-datetime")]
            DatetimeRange { .. } => "datetime_range",
            #[cfg(feature = "dtype-datetime")]
            DatetimeRanges { .. } => "datetime_ranges",
            #[cfg(feature = "dtype-time")]
            TimeRange { .. } => "time_range",
            #[cfg(feature = "dtype-time")]
            TimeRanges { .. } => "time_ranges",
        };
        write!(f, "{s}")
    }
}
