use arrow::temporal_conversions::{MICROSECONDS, MILLISECONDS, NANOSECONDS, SECONDS_IN_DAY};
use polars_core::datatypes::{AnyValue, DataType, TimeUnit};
use polars_core::prelude::Series;
use polars_error::PolarsResult;

pub fn impl_duration(s: &[Series], time_unit: TimeUnit) -> PolarsResult<Series> {
    if s.iter().any(|s| s.is_empty()) {
        return Ok(Series::new_empty(
            s[0].name(),
            &DataType::Duration(time_unit),
        ));
    }

    // TODO: Handle overflow for UInt64
    let weeks = s[0].cast(&DataType::Int64).unwrap();
    let days = s[1].cast(&DataType::Int64).unwrap();
    let hours = s[2].cast(&DataType::Int64).unwrap();
    let minutes = s[3].cast(&DataType::Int64).unwrap();
    let seconds = s[4].cast(&DataType::Int64).unwrap();
    let mut milliseconds = s[5].cast(&DataType::Int64).unwrap();
    let mut microseconds = s[6].cast(&DataType::Int64).unwrap();
    let mut nanoseconds = s[7].cast(&DataType::Int64).unwrap();

    let is_scalar = |s: &Series| s.len() == 1;
    let is_zero_scalar = |s: &Series| is_scalar(s) && s.get(0).unwrap() == AnyValue::Int64(0);

    // Process subseconds
    let max_len = s.iter().map(|s| s.len()).max().unwrap();
    let mut duration = match time_unit {
        TimeUnit::Microseconds => {
            if is_scalar(&microseconds) {
                microseconds = microseconds.new_from_index(0, max_len);
            }
            if !is_zero_scalar(&nanoseconds) {
                microseconds = (microseconds + (nanoseconds.wrapping_trunc_div_scalar(1_000)))?;
            }
            if !is_zero_scalar(&milliseconds) {
                microseconds = (microseconds + (milliseconds * 1_000))?;
            }
            microseconds
        },
        TimeUnit::Nanoseconds => {
            if is_scalar(&nanoseconds) {
                nanoseconds = nanoseconds.new_from_index(0, max_len);
            }
            if !is_zero_scalar(&microseconds) {
                nanoseconds = (nanoseconds + (microseconds * 1_000))?;
            }
            if !is_zero_scalar(&milliseconds) {
                nanoseconds = (nanoseconds + (milliseconds * 1_000_000))?;
            }
            nanoseconds
        },
        TimeUnit::Milliseconds => {
            if is_scalar(&milliseconds) {
                milliseconds = milliseconds.new_from_index(0, max_len);
            }
            if !is_zero_scalar(&nanoseconds) {
                milliseconds = (milliseconds + (nanoseconds.wrapping_trunc_div_scalar(1_000_000)))?;
            }
            if !is_zero_scalar(&microseconds) {
                milliseconds = (milliseconds + (microseconds.wrapping_trunc_div_scalar(1_000)))?;
            }
            milliseconds
        },
    };

    // Process other duration specifiers
    let multiplier = match time_unit {
        TimeUnit::Nanoseconds => NANOSECONDS,
        TimeUnit::Microseconds => MICROSECONDS,
        TimeUnit::Milliseconds => MILLISECONDS,
    };
    if !is_zero_scalar(&seconds) {
        duration = (duration + seconds * multiplier)?;
    }
    if !is_zero_scalar(&minutes) {
        duration = (duration + minutes * (multiplier * 60))?;
    }
    if !is_zero_scalar(&hours) {
        duration = (duration + hours * (multiplier * 60 * 60))?;
    }
    if !is_zero_scalar(&days) {
        duration = (duration + days * (multiplier * SECONDS_IN_DAY))?;
    }
    if !is_zero_scalar(&weeks) {
        duration = (duration + weeks * (multiplier * SECONDS_IN_DAY * 7))?;
    }

    duration.cast(&DataType::Duration(time_unit))
}
