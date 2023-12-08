#[cfg(feature = "temporal")]
mod date_range;
#[cfg(feature = "dtype-datetime")]
mod datetime_range;
mod int_range;
#[cfg(feature = "dtype-time")]
mod time_range;
mod utils;

use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
use polars_core::series::Series;
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
    IntRanges {
        step: i64,
    },
    #[cfg(feature = "temporal")]
    DateRange {
        interval: Duration,
        closed: ClosedWindow,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
    },
    #[cfg(feature = "temporal")]
    DateRanges {
        interval: Duration,
        closed: ClosedWindow,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
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
        let field = match self {
            IntRange { dtype, .. } => Field::new("int", dtype.clone()),
            IntRanges { .. } => Field::new("int_range", DataType::List(Box::new(DataType::Int64))),
            #[cfg(feature = "temporal")]
            DateRange {
                interval,
                closed: _,
                time_unit,
                time_zone,
            } => {
                // output dtype may change based on `interval`, `time_unit`, and `time_zone`
                let dtype = mapper.map_to_date_range_dtype(
                    interval,
                    time_unit.as_ref(),
                    time_zone.as_deref(),
                )?;
                return Ok(Field::new("date", dtype));
            },
            #[cfg(feature = "temporal")]
            DateRanges {
                interval,
                closed: _,
                time_unit,
                time_zone,
            } => {
                // output dtype may change based on `interval`, `time_unit`, and `time_zone`
                let inner_dtype = mapper.map_to_date_range_dtype(
                    interval,
                    time_unit.as_ref(),
                    time_zone.as_deref(),
                )?;
                return Ok(Field::new(
                    "date_range",
                    DataType::List(Box::new(inner_dtype)),
                ));
            },
            #[cfg(feature = "temporal")]
            DatetimeRange {
                interval: _,
                closed: _,
                time_unit,
                time_zone,
            } => {
                // output dtype may change based on `interval`, `time_unit`, and `time_zone`
                let dtype =
                    mapper.map_to_datetime_range_dtype(time_unit.as_ref(), time_zone.as_deref())?;
                return Ok(Field::new("datetime", dtype));
            },
            #[cfg(feature = "temporal")]
            DatetimeRanges {
                interval: _,
                closed: _,
                time_unit,
                time_zone,
            } => {
                // output dtype may change based on `interval`, `time_unit`, and `time_zone`
                let inner_dtype =
                    mapper.map_to_datetime_range_dtype(time_unit.as_ref(), time_zone.as_deref())?;
                return Ok(Field::new(
                    "datetime_range",
                    DataType::List(Box::new(inner_dtype)),
                ));
            },
            #[cfg(feature = "dtype-time")]
            TimeRange { .. } => {
                return Ok(Field::new("time", DataType::Time));
            },
            #[cfg(feature = "dtype-time")]
            TimeRanges { .. } => {
                return Ok(Field::new(
                    "time_range",
                    DataType::List(Box::new(DataType::Time)),
                ));
            },
        };
        Ok(field)
    }
}

impl Display for RangeFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RangeFunction::*;
        let s = match self {
            IntRange { .. } => "int_range",
            IntRanges { .. } => "int_ranges",
            #[cfg(feature = "temporal")]
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
            IntRanges { step } => {
                map_as_slice!(int_range::int_ranges, step)
            },
            #[cfg(feature = "temporal")]
            DateRange {
                interval,
                closed,
                time_unit,
                time_zone,
            } => {
                map_as_slice!(
                    date_range::temporal_range,
                    interval,
                    closed,
                    time_unit,
                    time_zone.clone()
                )
            },
            #[cfg(feature = "temporal")]
            DateRanges {
                interval,
                closed,
                time_unit,
                time_zone,
            } => {
                map_as_slice!(
                    date_range::temporal_ranges,
                    interval,
                    closed,
                    time_unit,
                    time_zone.clone()
                )
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
