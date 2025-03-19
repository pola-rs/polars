use arrow::temporal_conversions::MILLISECONDS_IN_DAY;
use polars_core::prelude::*;
use polars_ops::series::{ClosedInterval, new_linear_space_f32, new_linear_space_f64};

use super::utils::{build_nulls, ensure_range_bounds_contain_exactly_one_value};

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

pub(super) fn linear_spaces(
    s: &[Column],
    closed: ClosedInterval,
    array_width: Option<usize>,
) -> PolarsResult<Column> {
    let start = &s[0];
    let end = &s[1];

    let (num_samples, capacity_factor) = match array_width {
        Some(ns) => {
            // An array width is provided instead of a column of `num_sample`s.
            let scalar = Scalar::new(DataType::UInt64, AnyValue::UInt64(ns as u64));
            (&Column::new_scalar(PlSmallStr::EMPTY, scalar, 1), ns)
        },
        None => (&s[2], CAPACITY_FACTOR),
    };
    let name = start.name().clone();

    let num_samples = num_samples.strict_cast(&DataType::UInt64)?;
    let num_samples = num_samples.u64()?;
    let len = start.len().max(end.len()).max(num_samples.len());

    match (start.dtype(), end.dtype()) {
        (DataType::Float32, DataType::Float32) => {
            let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
                name,
                len,
                len * capacity_factor,
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

            let to_type = array_width.map_or_else(
                || DataType::List(Box::new(DataType::Float32)),
                |width| DataType::Array(Box::new(DataType::Float32), width),
            );
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
                len * capacity_factor,
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

            let to_type = array_width.map_or_else(
                || DataType::List(Box::new(dt.clone())),
                |width| DataType::Array(Box::new(dt.clone()), width),
            );
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
                len * capacity_factor,
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

            let to_type = array_width.map_or_else(
                || DataType::List(Box::new(DataType::Float64)),
                |width| DataType::Array(Box::new(DataType::Float64), width),
            );
            out.cast(&to_type)
        },
    }
}

/// Create a ranges column from the given start/end columns and a range function.
pub(super) fn linear_spaces_impl_broadcast<T, F>(
    start: &ChunkedArray<T>,
    end: &ChunkedArray<T>,
    num_samples: &UInt64Chunked,
    linear_space_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<T>,
) -> PolarsResult<Column>
where
    T: PolarsFloatType,
    F: Fn(T::Native, T::Native, u64, &mut ListPrimitiveChunkedBuilder<T>) -> PolarsResult<()>,
    ListPrimitiveChunkedBuilder<T>: ListBuilderTrait,
{
    match (start.len(), end.len(), num_samples.len()) {
        (len_start, len_end, len_samples) if len_start == len_end && len_start == len_samples => {
            // (n, n, n)
            build_linear_spaces::<_, _, _, T, F>(
                start.iter(),
                end.iter(),
                num_samples.iter(),
                linear_space_impl,
                builder,
            )?;
        },
        // (1, n, n)
        (1, len_end, len_samples) if len_end == len_samples => {
            let start_value = start.get(0);
            if start_value.is_some() {
                build_linear_spaces::<_, _, _, T, F>(
                    std::iter::repeat(start_value),
                    end.iter(),
                    num_samples.iter(),
                    linear_space_impl,
                    builder,
                )?
            } else {
                build_nulls(builder, len_end)
            }
        },
        // (n, 1, n)
        (len_start, 1, len_samples) if len_start == len_samples => {
            let end_value = end.get(0);
            if end_value.is_some() {
                build_linear_spaces::<_, _, _, T, F>(
                    start.iter(),
                    std::iter::repeat(end_value),
                    num_samples.iter(),
                    linear_space_impl,
                    builder,
                )?
            } else {
                build_nulls(builder, len_start)
            }
        },
        // (n, n, 1)
        (len_start, len_end, 1) if len_start == len_end => {
            let num_samples_value = num_samples.get(0);
            if num_samples_value.is_some() {
                build_linear_spaces::<_, _, _, T, F>(
                    start.iter(),
                    end.iter(),
                    std::iter::repeat(num_samples_value),
                    linear_space_impl,
                    builder,
                )?
            } else {
                build_nulls(builder, len_start)
            }
        },
        // (n, 1, 1)
        (len_start, 1, 1) => {
            let end_value = end.get(0);
            let num_samples_value = num_samples.get(0);
            match (end_value, num_samples_value) {
                (Some(_), Some(_)) => build_linear_spaces::<_, _, _, T, F>(
                    start.iter(),
                    std::iter::repeat(end_value),
                    std::iter::repeat(num_samples_value),
                    linear_space_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len_start),
            }
        },
        // (1, n, 1)
        (1, len_end, 1) => {
            let start_value = start.get(0);
            let num_samples_value = num_samples.get(0);
            match (start_value, num_samples_value) {
                (Some(_), Some(_)) => build_linear_spaces::<_, _, _, T, F>(
                    std::iter::repeat(start_value),
                    end.iter(),
                    std::iter::repeat(num_samples_value),
                    linear_space_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len_end),
            }
        },
        // (1, 1, n)
        (1, 1, len_num_samples) => {
            let start_value = start.get(0);
            let end_value = end.get(0);
            match (start_value, end_value) {
                (Some(_), Some(_)) => build_linear_spaces::<_, _, _, T, F>(
                    std::iter::repeat(start_value),
                    std::iter::repeat(end_value),
                    num_samples.iter(),
                    linear_space_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len_num_samples),
            }
        },
        (len_start, len_end, len_num_samples) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}), `end` ({}), and `num_samples` ({}) do not match",
                len_start, len_end, len_num_samples
            )
        },
    };
    let out = builder.finish().into_column();
    Ok(out)
}

/// Iterate over a start and end column and create a range for each entry.
fn build_linear_spaces<I, J, K, T, F>(
    start: I,
    end: J,
    num_samples: K,
    linear_space_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<T>,
) -> PolarsResult<()>
where
    I: Iterator<Item = Option<T::Native>>,
    J: Iterator<Item = Option<T::Native>>,
    K: Iterator<Item = Option<u64>>,
    T: PolarsFloatType,
    F: Fn(T::Native, T::Native, u64, &mut ListPrimitiveChunkedBuilder<T>) -> PolarsResult<()>,
    ListPrimitiveChunkedBuilder<T>: ListBuilderTrait,
{
    for ((start, end), num_samples) in start.zip(end).zip(num_samples) {
        match (start, end, num_samples) {
            (Some(start), Some(end), Some(num_samples)) => {
                linear_space_impl(start, end, num_samples, builder)?
            },
            _ => builder.append_null(),
        }
    }
    Ok(())
}
