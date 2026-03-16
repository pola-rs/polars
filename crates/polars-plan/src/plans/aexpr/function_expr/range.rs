use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
use polars_ops::series::ClosedInterval;
#[cfg(feature = "temporal")]
use polars_time::{ClosedWindow, Duration};

use super::{FunctionOptions, IRFunctionExpr};
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
use crate::dsl::function_expr::DateRangeArgs;
use crate::plans::aexpr::function_expr::FieldsMapper;
use crate::prelude::FunctionFlags;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRRangeFunction {
    IntRange {
        step: i64,
        dtype: DataType,
    },
    IntRanges {
        dtype: DataType,
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

fn map_linspace_dtype(mapper: &FieldsMapper) -> PolarsResult<DataType> {
    let fields = mapper.args();
    let start_dtype = fields[0].dtype();
    let end_dtype = fields[1].dtype();
    Ok(match (start_dtype, end_dtype) {
        #[cfg(feature = "dtype-f16")]
        (&DataType::Float16, &DataType::Float16) => DataType::Float16,
        (&DataType::Float32, &DataType::Float32) => DataType::Float32,
        // A linear space of a Date produces a sequence of Datetimes
        (dt1, dt2) if dt1.is_temporal() && dt1 == dt2 => {
            if dt1 == &DataType::Date {
                DataType::Datetime(TimeUnit::Microseconds, None)
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

impl IRRangeFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRRangeFunction::*;
        match self {
            IntRange { dtype, .. } => mapper.with_dtype(dtype.clone()),
            IntRanges { dtype } => mapper.with_dtype(DataType::List(Box::new(dtype.clone()))),
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
            DateRange {
                interval: _,
                closed: _,
                arg_type: _,
            } => mapper.with_dtype(DataType::Date),
            #[cfg(feature = "dtype-date")]
            DateRanges {
                interval: _,
                closed: _,
                arg_type: _,
            } => mapper.with_dtype(DataType::List(Box::new(DataType::Date))),
            #[cfg(feature = "dtype-datetime")]
            DatetimeRange {
                interval: _,
                closed: _,
                time_unit,
                time_zone,
                arg_type: _,
            } => {
                // Output dtype may change based on `interval`, `time_unit`, and `time_zone`.
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
                arg_type: _,
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
        use IRRangeFunction as R;
        match self {
            R::IntRange { .. } => {
                FunctionOptions::groupwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            R::IntRanges { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            R::LinearSpace { .. } => {
                FunctionOptions::groupwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            R::LinearSpaces { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-date")]
            R::DateRange { .. } => {
                FunctionOptions::groupwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-date")]
            R::DateRanges { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-datetime")]
            R::DatetimeRange { .. } => {
                FunctionOptions::groupwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-datetime")]
            R::DatetimeRanges { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-time")]
            R::TimeRange { .. } => {
                FunctionOptions::groupwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "dtype-time")]
            R::TimeRanges { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
        }
    }
}

impl Display for IRRangeFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRRangeFunction::*;
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

impl From<IRRangeFunction> for IRFunctionExpr {
    fn from(value: IRRangeFunction) -> Self {
        Self::Range(value)
    }
}

impl FieldsMapper<'_> {
    pub fn map_to_datetime_range_dtype(
        &self,
        time_unit: Option<&TimeUnit>,
        time_zone: Option<&TimeZone>,
    ) -> PolarsResult<DataType> {
        let data_dtype = self.map_to_supertype()?.dtype;

        let (data_tu, data_tz) = if let DataType::Datetime(tu, tz) = data_dtype {
            (tu, tz)
        } else {
            (TimeUnit::Microseconds, None)
        };

        let tu = match time_unit {
            Some(tu) => *tu,
            None => data_tu,
        };

        let tz = time_zone.cloned().or(data_tz);

        Ok(DataType::Datetime(tu, tz))
    }
}
