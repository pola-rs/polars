#[cfg(feature = "dtype-date")]
mod date_range;
#[cfg(feature = "dtype-datetime")]
mod datetime_range;
mod int_range;
mod linear_space;
#[cfg(feature = "dtype-time")]
mod time_range;
mod utils;

use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
use polars_ops::series::ClosedInterval;
#[cfg(feature = "temporal")]
use polars_time::{ClosedWindow, Duration};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{FunctionExpr, FunctionOptions};
use crate::dsl::SpecialEq;
use crate::dsl::function_expr::FieldsMapper;
use crate::map_as_slice;
use crate::prelude::{ColumnsUdf, FunctionFlags};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum RangeFunction {
    IntRange {
        step: i64,
        dtype: DataType,
    },
    IntRanges,
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

fn map_linspace_dtype(mapper: &FieldsMapper) -> PolarsResult<DataType> {
    let fields = mapper.args();
    let start_dtype = fields[0].dtype();
    let end_dtype = fields[1].dtype();
    Ok(match (start_dtype, end_dtype) {
        (&DataType::Float32, &DataType::Float32) => DataType::Float32,
        // A linear space of a Date produces a sequence of Datetimes
        (dt1, dt2) if dt1.is_temporal() && dt1 == dt2 => {
            if dt1 == &DataType::Date {
                DataType::Datetime(TimeUnit::Milliseconds, None)
            } else {
                dt1.clone()
            }
        },
        (dt1, dt2) if !dt1.is_primitive_numeric() || !dt2.is_primitive_numeric() => {
            polars_bail!(ComputeError:
                "'start' and 'end' have incompatible dtypes, got {:?} and {:?}",
                dt1, dt2
            )
        },
        _ => DataType::Float64,
    })
}

impl RangeFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use RangeFunction::*;
        match self {
            IntRange { dtype, .. } => mapper.with_dtype(dtype.clone()),
            IntRanges => mapper.with_dtype(DataType::List(Box::new(DataType::Int64))),
            LinearSpace { .. } => mapper.with_dtype(map_linspace_dtype(&mapper)?),
            LinearSpaces {
                closed: _,
                array_width,
            } => {
                let inner = Box::new(map_linspace_dtype(&mapper)?);
                let dt = match array_width {
                    Some(width) => DataType::Array(inner, *width),
                    None => DataType::List(inner),
                };
                mapper.with_dtype(dt)
            },
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
                    mapper.map_to_datetime_range_dtype(time_unit.as_ref(), time_zone.as_ref())?;
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
                    mapper.map_to_datetime_range_dtype(time_unit.as_ref(), time_zone.as_ref())?;
                mapper.with_dtype(DataType::List(Box::new(inner_dtype)))
            },
            #[cfg(feature = "dtype-time")]
            TimeRange { .. } => mapper.with_dtype(DataType::Time),
            #[cfg(feature = "dtype-time")]
            TimeRanges { .. } => mapper.with_dtype(DataType::List(Box::new(DataType::Time))),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use RangeFunction as R;
        match self {
            R::IntRange { .. } => {
                FunctionOptions::row_separable().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            R::LinearSpace { .. } => {
                FunctionOptions::row_separable().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-date")]
            R::DateRange { .. } => {
                FunctionOptions::row_separable().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-datetime")]
            R::DatetimeRange { .. } => FunctionOptions::row_separable()
                .with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
                .with_supertyping(Default::default()),
            #[cfg(feature = "dtype-time")]
            R::TimeRange { .. } => {
                FunctionOptions::row_separable().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            R::IntRanges => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            R::LinearSpaces { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-date")]
            R::DateRanges { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-datetime")]
            R::DatetimeRanges { .. } => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
                .with_supertyping(Default::default()),
            #[cfg(feature = "dtype-time")]
            R::TimeRanges { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
        }
    }
}

impl Display for RangeFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RangeFunction::*;
        let s = match self {
            IntRange { .. } => "int_range",
            IntRanges => "int_ranges",
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

impl From<RangeFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: RangeFunction) -> Self {
        use RangeFunction::*;
        match func {
            IntRange { step, dtype } => {
                map_as_slice!(int_range::int_range, step, dtype.clone())
            },
            IntRanges => {
                map_as_slice!(int_range::int_ranges)
            },
            LinearSpace { closed } => {
                map_as_slice!(linear_space::linear_space, closed)
            },
            LinearSpaces {
                closed,
                array_width,
            } => {
                map_as_slice!(linear_space::linear_spaces, closed, array_width)
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

impl From<RangeFunction> for FunctionExpr {
    fn from(value: RangeFunction) -> Self {
        Self::Range(value)
    }
}
