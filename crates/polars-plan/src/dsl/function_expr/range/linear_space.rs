use arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_core::prelude::*;
use polars_ops::series::{new_linear_space_f32, new_linear_space_f64, ClosedInterval};

use super::utils::ensure_range_bounds_contain_exactly_one_value;

pub(super) fn linear_space(s: &[Column], closed: ClosedInterval) -> PolarsResult<Column> {
    let start = &s[0];
    let end = &s[1];
    let num_samples = &s[2];
    let name = start.name();

    ensure_range_bounds_contain_exactly_one_value(start, end)?;
    polars_ensure!(
        num_samples.len() == 1,
        ComputeError: "`num_samples` must contain exactly one value, got {} values", num_samples.len()
    );

    let start = start.get(0).unwrap();
    let end = end.get(0).unwrap();
    let num_samples = num_samples.get(0).unwrap();
    let num_samples = num_samples
        .extract::<u64>()
        .ok_or(PolarsError::ComputeError(
            format!(
                "'num_samples' must be non-negative integer, got {}",
                num_samples
            )
            .into(),
        ))?;

    match (start.dtype(), end.dtype()) {
        (DataType::Float32, DataType::Float32) => new_linear_space_f32(
            start.extract::<f32>().unwrap(),
            end.extract::<f32>().unwrap(),
            num_samples,
            closed,
            name.clone(),
        ),
        (mut dt, dt2) if dt.is_temporal() && dt == dt2 => {
            let mut start = start.extract::<i64>().unwrap();
            let mut end = end.extract::<i64>().unwrap();

            // A linear space of a Date produces a sequence of Datetimes, so we must upcast.
            if dt == DataType::Date {
                start *= MILLISECONDS_IN_DAY;
                end *= MILLISECONDS_IN_DAY;
                dt = DataType::Datetime(TimeUnit::Milliseconds, None);
            }
            new_linear_space_f64(start as f64, end as f64, num_samples, closed, name.clone())
                .map(|s| s.cast(&dt).unwrap())
        },
        (dt1, dt2) if !dt1.is_primitive_numeric() || !dt2.is_primitive_numeric() => {
            Err(PolarsError::ComputeError(
                format!(
                    "'start' and 'end' have incompatible dtypes, got {:?} and {:?}",
                    dt1, dt2
                )
                .into(),
            ))
        },
        (_, _) => new_linear_space_f64(
            start.extract::<f64>().unwrap(),
            end.extract::<f64>().unwrap(),
            num_samples,
            closed,
            name.clone(),
        ),
    }
    .map(Column::from)
}
