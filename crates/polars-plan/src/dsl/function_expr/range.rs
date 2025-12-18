use std::fmt;

use polars_core::prelude::*;
use polars_ops::series::ClosedInterval;
#[cfg(feature = "temporal")]
use polars_time::{ClosedWindow, Duration};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{DataTypeExpr, Expr, FunctionExpr};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
// Date and Datetime range functions requires three of four optional parameters.
// The combination of parameters used determines the dtype and usage of the input expressions.
pub enum DateRangeArgs {
    StartEndInterval,
    StartEndSamples,
    StartIntervalSamples,
    EndIntervalSamples,
}

impl DateRangeArgs {
    pub fn parse(
        start: Option<Expr>,
        end: Option<Expr>,
        interval: Option<Duration>,
        num_samples: Option<Expr>,
    ) -> PolarsResult<(Vec<Expr>, DateRangeArgs)> {
        match (start, end, interval, num_samples) {
            (Some(start), Some(end), Some(_), None) => {
                Ok((vec![start, end], DateRangeArgs::StartEndInterval))
            },
            (Some(start), Some(end), None, Some(num_samples)) => Ok((
                vec![start, end, num_samples],
                DateRangeArgs::StartEndSamples,
            )),
            (Some(start), None, Some(_), Some(num_samples)) => Ok((
                vec![start, num_samples],
                DateRangeArgs::StartIntervalSamples,
            )),
            (None, Some(end), Some(_), Some(num_samples)) => {
                Ok((vec![end, num_samples], DateRangeArgs::EndIntervalSamples))
            },
            _ => {
                polars_bail!(InvalidOperation: "Exactly three of 'start', 'end', 'interval', and 'num_samples' must be supplied.");
            },
        }
    }
}

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
        interval: Option<Duration>,
        closed: ClosedWindow,
        arg_type: DateRangeArgs,
    },
    #[cfg(feature = "dtype-date")]
    DateRanges {
        interval: Option<Duration>,
        closed: ClosedWindow,
        arg_type: DateRangeArgs,
    },
    #[cfg(feature = "dtype-datetime")]
    DatetimeRange {
        interval: Option<Duration>,
        closed: ClosedWindow,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
        arg_type: DateRangeArgs,
    },
    #[cfg(feature = "dtype-datetime")]
    DatetimeRanges {
        interval: Option<Duration>,
        closed: ClosedWindow,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
        arg_type: DateRangeArgs,
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
            #[cfg(feature = "dtype-date")]
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
