use std::sync::Arc;

use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRRangeFunction;

#[cfg(feature = "dtype-datetime")]
mod datetime_range;
mod int_range;
mod linear_space;
#[cfg(feature = "dtype-time")]
mod time_range;
mod utils;

pub fn function_expr_to_udf(func: IRRangeFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRRangeFunction::*;
    match func {
        IntRange { step, dtype } => {
            map_as_slice!(int_range::int_range, step, dtype.clone())
        },
        IntRanges { dtype } => {
            map_as_slice!(int_range::int_ranges, dtype.clone())
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
        DateRange {
            interval,
            closed,
            arg_type,
        } => {
            map_as_slice!(datetime_range::date_range, interval, closed, arg_type) // TODO! num_samples
        },
        #[cfg(feature = "dtype-date")]
        DateRanges {
            interval,
            closed,
            arg_type,
        } => {
            map_as_slice!(datetime_range::date_ranges, interval, closed, arg_type) // TODO! num_samples
        },
        #[cfg(feature = "dtype-datetime")]
        DatetimeRange {
            interval,
            closed,
            time_unit: _,
            time_zone: _,
            arg_type,
        } => {
            map_as_slice!(datetime_range::datetime_range, interval, closed, arg_type) // TODO! num_samples
        },
        #[cfg(feature = "dtype-datetime")]
        DatetimeRanges {
            interval,
            closed,
            time_unit: _,
            time_zone: _,
            arg_type,
        } => {
            map_as_slice!(datetime_range::datetime_ranges, interval, closed, arg_type) // TODO! num_samples
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
