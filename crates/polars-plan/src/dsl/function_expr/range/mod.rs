#[cfg(feature = "dtype-date")]
mod date_range;
#[cfg(feature = "dtype-datetime")]
mod datetime_range;
mod int_range;
#[cfg(feature = "dtype-time")]
mod time_range;
mod utils;

use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
#[cfg(feature = "temporal")]
use polars_time::{ClosedWindow, Duration};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::dsl::function_expr::FieldsMapper;
use crate::dsl::SpecialEq;
use crate::map_as_slice;
use crate::prelude::SeriesUdf;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum RangeFunction {
    IntRange {
        step: i64,
        dtype: DataType,
    },
    IntRanges,
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

impl RangeFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use RangeFunction::*;
        match self {
            IntRange { dtype, .. } => mapper.with_dtype(dtype.clone()),
            IntRanges => mapper.with_dtype(DataType::List(Box::new(DataType::Int64))),
            #[cfg(feature = "dtype-date")]
            DateRange { .. } => mapper.with_dtype(DataType::Date),
            #[cfg(feature = "dtype-date")]
            DateRanges { .. } => mapper.with_dtype(DataType::List(Box::new(DataType::Date))),
            #[cfg(feature = "dtype-datetime")]
            DatetimeRange {
                interval: _,
                closed: _,
                time_unit,
                time_zone,
            } => {
                // output dtype may change based on `interval`, `time_unit`, and `time_zone`
                let dtype =
                    mapper.map_to_datetime_range_dtype(time_unit.as_ref(), time_zone.as_deref())?;
                mapper.with_dtype(dtype)
            },
            #[cfg(feature = "dtype-datetime")]
            DatetimeRanges {
                interval: _,
                closed: _,
                time_unit,
                time_zone,
            } => {
                // output dtype may change based on `interval`, `time_unit`, and `time_zone`
                let inner_dtype =
                    mapper.map_to_datetime_range_dtype(time_unit.as_ref(), time_zone.as_deref())?;
                mapper.with_dtype(DataType::List(Box::new(inner_dtype)))
            },
            #[cfg(feature = "dtype-time")]
            TimeRange { .. } => mapper.with_dtype(DataType::Time),
            #[cfg(feature = "dtype-time")]
            TimeRanges { .. } => mapper.with_dtype(DataType::List(Box::new(DataType::Time))),
        }
    }
}

impl Display for RangeFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RangeFunction::*;
        let s = match self {
            IntRange { .. } => "int_range",
            IntRanges => "int_ranges",
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

impl From<RangeFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: RangeFunction) -> Self {
        use RangeFunction::*;
        match func {
            IntRange { step, dtype } => {
                map_as_slice!(int_range::int_range, step, dtype.clone())
            },
            IntRanges => {
                map_as_slice!(int_range::int_ranges)
            },
            #[cfg(feature = "dtype-date")]
            DateRange { interval, closed } => {
                map_as_slice!(date_range::date_range, interval, closed)
            },
            #[cfg(feature = "dtype-date")]
            DateRanges { interval, closed } => {
                map_as_slice!(date_range::date_ranges, interval, closed)
            },
            #[cfg(feature = "dtype-datetime")]
            DatetimeRange {
                interval,
                closed,
                time_unit,
                time_zone,
            } => {
                map_as_slice!(
                    datetime_range::datetime_range,
                    interval,
                    closed,
                    time_unit,
                    time_zone.clone()
                )
            },
            #[cfg(feature = "dtype-datetime")]
            DatetimeRanges {
                interval,
                closed,
                time_unit,
                time_zone,
            } => {
                map_as_slice!(
                    datetime_range::datetime_ranges,
                    interval,
                    closed,
                    time_unit,
                    time_zone.clone()
                )
            },
            #[cfg(feature = "dtype-time")]
            TimeRange { interval, closed } => {
                map_as_slice!(time_range::time_range, interval, closed)
            },
            #[cfg(feature = "dtype-time")]
            TimeRanges { interval, closed } => {
                map_as_slice!(time_range::time_ranges, interval, closed)
            },
        }
    }
}
