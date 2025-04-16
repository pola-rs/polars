use arrow::temporal_conversions::{MICROSECONDS, MILLISECONDS, NANOSECONDS, SECONDS_IN_DAY};
use polars_core::datatypes::{AnyValue, DataType, TimeUnit};
use polars_core::prelude::Column;
use polars_error::PolarsResult;

pub fn impl_duration(s: &[Column], time_unit: TimeUnit) -> PolarsResult<Column> {
    if s.iter().any(|s| s.is_empty()) {
        return Ok(Column::new_empty(
            s[0].name().clone(),
            &DataType::Duration(time_unit),
        ));
    }

    // TODO: Handle overflow for UInt64
    let weeks = &s[0];
    let days = &s[1];
    let hours = &s[2];
    let minutes = &s[3];
    let seconds = &s[4];
    let milliseconds = &s[5];
    let microseconds = &s[6];
    let nanoseconds = &s[7];

    let is_scalar = |s: &Column| s.len() == 1;
    let is_zero_scalar =
        |s: &Column| is_scalar(s) && s.get(0).unwrap() == AnyValue::zero_sum(s.dtype());

    // Process subseconds
    let max_len = s.iter().map(|s| s.len()).max().unwrap();
    let mut duration = match time_unit {
        TimeUnit::Microseconds => {
            let mut duration = microseconds.cast(&DataType::Int64).unwrap();
            if is_scalar(&duration) {
                duration = duration.new_from_index(0, max_len);
            }
            if !is_zero_scalar(&nanoseconds) {
                duration = (duration
                    + nanoseconds
                        .wrapping_trunc_div_scalar(1_000)
                        .cast(&DataType::Int64)
                        .unwrap())?;
            }
            if !is_zero_scalar(&milliseconds) {
                duration = (duration + (milliseconds * 1_000).cast(&DataType::Int64).unwrap())?;
            }
            duration
        },
        TimeUnit::Nanoseconds => {
            let mut duration = nanoseconds.cast(&DataType::Int64).unwrap();
            if is_scalar(&nanoseconds) {
                duration = duration.new_from_index(0, max_len);
            }
            if !is_zero_scalar(&microseconds) {
                duration = (duration + (microseconds * 1_000).cast(&DataType::Int64).unwrap())?;
            }
            if !is_zero_scalar(&milliseconds) {
                duration = (duration + (milliseconds * 1_000_000).cast(&DataType::Int64).unwrap())?;
            }
            duration
        },
        TimeUnit::Milliseconds => {
            let mut duration = milliseconds.cast(&DataType::Int64).unwrap();
            if is_scalar(&milliseconds) {
                duration = duration.new_from_index(0, max_len);
            }
            if !is_zero_scalar(&nanoseconds) {
                duration = (duration
                    + nanoseconds
                        .wrapping_trunc_div_scalar(1_000_000)
                        .cast(&DataType::Int64)
                        .unwrap())?;
            }
            if !is_zero_scalar(&microseconds) {
                duration = (duration
                    + microseconds
                        .wrapping_trunc_div_scalar(1_000)
                        .cast(&DataType::Int64)
                        .unwrap())?;
            }
            duration
        },
    };

    // Process other duration specifiers
    let multiplier = match time_unit {
        TimeUnit::Nanoseconds => NANOSECONDS,
        TimeUnit::Microseconds => MICROSECONDS,
        TimeUnit::Milliseconds => MILLISECONDS,
    };
    if !is_zero_scalar(&seconds) {
        duration = (duration + (seconds * multiplier).cast(&DataType::Int64).unwrap())?;
    }
    if !is_zero_scalar(&minutes) {
        duration = (duration + (minutes * multiplier * 60).cast(&DataType::Int64).unwrap())?;
    }
    if !is_zero_scalar(&hours) {
        duration = (duration
            + (hours * multiplier * 60 * 60)
                .cast(&DataType::Int64)
                .unwrap())?;
    }
    if !is_zero_scalar(&days) {
        duration = (duration
            + (days * multiplier * SECONDS_IN_DAY)
                .cast(&DataType::Int64)
                .unwrap())?;
    }
    if !is_zero_scalar(&weeks) {
        duration = (duration
            + (weeks * multiplier * SECONDS_IN_DAY * 7)
                .cast(&DataType::Int64)
                .unwrap())?;
    }

    duration.cast(&DataType::Duration(time_unit))
}
