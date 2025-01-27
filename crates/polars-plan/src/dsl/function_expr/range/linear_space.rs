use arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_core::prelude::*;
use polars_ops::series::{new_linear_space_f32, new_linear_space_f64, ClosedInterval};

use super::utils::{ensure_range_bounds_contain_exactly_one_value, linear_spaces_impl_broadcast};

const CAPACITY_FACTOR: usize = 5;

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
        )
        .map(|s| s.into_column()),
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
                .map(|s| s.cast(&dt).unwrap().into_column())
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
        )
        .map(|s| s.into_column()),
    }
}

pub(super) fn linear_spaces(s: &[Column], closed: ClosedInterval) -> PolarsResult<Column> {
    let start = &s[0];
    let end = &s[1];
    let num_samples = &s[2];
    let name = start.name().clone();

    let num_samples = num_samples.strict_cast(&DataType::UInt64)?;
    let num_samples = num_samples.u64()?;
    let len = start.len().max(end.len()).max(num_samples.len());

    match (start.dtype(), end.dtype()) {
        (DataType::Float32, DataType::Float32) => {
            let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
                name,
                len,
                len * CAPACITY_FACTOR,
                DataType::Float32,
            );

            let linspace_impl =
                |start,
                 end,
                 num_samples,
                 builder: &mut ListPrimitiveChunkedBuilder<Float32Type>| {
                    let ls =
                        new_linear_space_f32(start, end, num_samples, closed, PlSmallStr::EMPTY)?;
                    builder.append_slice(ls.cont_slice().unwrap());
                    Ok(())
                };

            let start = start.f32()?;
            let end = end.f32()?;
            let out =
                linear_spaces_impl_broadcast(start, end, num_samples, linspace_impl, &mut builder)?;

            let to_type = DataType::List(Box::new(DataType::Float32));
            out.cast(&to_type)
        },
        (mut dt, dt2) if dt.is_temporal() && dt == dt2 => {
            let mut start = start.to_physical_repr();
            let mut end = end.to_physical_repr();

            // A linear space of a Date produces a sequence of Datetimes, so we must upcast.
            if dt == &DataType::Date {
                start = start.cast(&DataType::Int64)? * MILLISECONDS_IN_DAY;
                end = end.cast(&DataType::Int64)? * MILLISECONDS_IN_DAY;
                dt = &DataType::Datetime(TimeUnit::Milliseconds, None);
            }

            let start = start.cast(&DataType::Float64)?;
            let start = start.f64()?;
            let end = end.cast(&DataType::Float64)?;
            let end = end.f64()?;

            let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                name,
                len,
                len * CAPACITY_FACTOR,
                DataType::Float64,
            );

            let linspace_impl =
                |start,
                 end,
                 num_samples,
                 builder: &mut ListPrimitiveChunkedBuilder<Float64Type>| {
                    let ls =
                        new_linear_space_f64(start, end, num_samples, closed, PlSmallStr::EMPTY)?;
                    builder.append_slice(ls.cont_slice().unwrap());
                    Ok(())
                };
            let out =
                linear_spaces_impl_broadcast(start, end, num_samples, linspace_impl, &mut builder)?;

            let to_type = DataType::List(Box::new(dt.clone()));
            out.cast(&to_type)
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
        (_, _) => {
            let start = start.strict_cast(&DataType::Float64)?;
            let end = end.strict_cast(&DataType::Float64)?;
            let start = start.f64()?;
            let end = end.f64()?;

            let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                name,
                len,
                len * CAPACITY_FACTOR,
                DataType::Float64,
            );

            let linspace_impl =
                |start,
                 end,
                 num_samples,
                 builder: &mut ListPrimitiveChunkedBuilder<Float64Type>| {
                    let ls =
                        new_linear_space_f64(start, end, num_samples, closed, PlSmallStr::EMPTY)?;
                    builder.append_slice(ls.cont_slice().unwrap());
                    Ok(())
                };
            let out =
                linear_spaces_impl_broadcast(start, end, num_samples, linspace_impl, &mut builder)?;

            let to_type = DataType::List(Box::new(DataType::Float64));
            out.cast(&to_type)
        },
    }
}
