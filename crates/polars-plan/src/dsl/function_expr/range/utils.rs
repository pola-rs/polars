use polars_core::prelude::{
    polars_bail, polars_ensure, ChunkedArray, Int64Chunked, IntoSeries, ListBuilderTrait,
    ListPrimitiveChunkedBuilder, PolarsIntegerType, PolarsResult, Series,
};

pub(super) fn temporal_series_to_i64_scalar(s: &Series) -> Option<i64> {
    s.to_physical_repr().get(0).unwrap().extract::<i64>()
}
pub(super) fn ensure_range_bounds_contain_exactly_one_value(
    start: &Series,
    end: &Series,
) -> PolarsResult<()> {
    polars_ensure!(
        start.len() == 1,
        ComputeError: "`start` must contain exactly one value, got {} values", start.len()
    );
    polars_ensure!(
        end.len() == 1,
        ComputeError: "`end` must contain exactly one value, got {} values", end.len()
    );
    Ok(())
}

/// Create a numeric ranges column from the given start/end/step columns and a range function.
pub(super) fn numeric_ranges_impl_broadcast<T, U, F>(
    start: &ChunkedArray<T>,
    end: &ChunkedArray<T>,
    step: &Int64Chunked,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<Series>
where
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, i64, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
{
    match (start.len(), end.len(), step.len()) {
        (len_start, len_end, len_step) if len_start == len_end && len_start == len_step => {
            build_numeric_ranges::<_, _, _, T, U, F>(
                start.downcast_iter().flatten(),
                end.downcast_iter().flatten(),
                step.downcast_iter().flatten(),
                range_impl,
                builder,
            )?;
        },
        (1, len_end, 1) => {
            let start_scalar = start.get(0);
            let step_scalar = step.get(0);
            match (start_scalar, step_scalar) {
                (Some(start), Some(step)) => build_numeric_ranges::<_, _, _, T, U, F>(
                    std::iter::repeat(Some(&start)),
                    end.downcast_iter().flatten(),
                    std::iter::repeat(Some(&step)),
                    range_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len_end),
            }
        },
        (len_start, 1, 1) => {
            let end_scalar = end.get(0);
            let step_scalar = step.get(0);
            match (end_scalar, step_scalar) {
                (Some(end), Some(step)) => build_numeric_ranges::<_, _, _, T, U, F>(
                    start.downcast_iter().flatten(),
                    std::iter::repeat(Some(&end)),
                    std::iter::repeat(Some(&step)),
                    range_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len_start),
            }
        },
        (1, 1, len_step) => {
            let start_scalar = start.get(0);
            let end_scalar = end.get(0);
            match (start_scalar, end_scalar) {
                (Some(start), Some(end)) => build_numeric_ranges::<_, _, _, T, U, F>(
                    std::iter::repeat(Some(&start)),
                    std::iter::repeat(Some(&end)),
                    step.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len_step),
            }
        },
        (len_start, len_end, 1) if len_start == len_end => {
            let step_scalar = step.get(0);
            match step_scalar {
                Some(step) => build_numeric_ranges::<_, _, _, T, U, F>(
                    start.downcast_iter().flatten(),
                    end.downcast_iter().flatten(),
                    std::iter::repeat(Some(&step)),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_start),
            }
        },
        (len_start, 1, len_step) if len_start == len_step => {
            let end_scalar = end.get(0);
            match end_scalar {
                Some(end) => build_numeric_ranges::<_, _, _, T, U, F>(
                    start.downcast_iter().flatten(),
                    std::iter::repeat(Some(&end)),
                    step.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_start),
            }
        },
        (1, len_end, len_step) if len_end == len_step => {
            let start_scalar = start.get(0);
            match start_scalar {
                Some(start) => build_numeric_ranges::<_, _, _, T, U, F>(
                    std::iter::repeat(Some(&start)),
                    end.downcast_iter().flatten(),
                    step.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_end),
            }
        },
        (len_start, len_end, len_step) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}), `end` ({}) and `step` ({}) do not match",
                len_start, len_end, len_step
            )
        },
    };
    let out = builder.finish().into_series();
    Ok(out)
}

/// Create a ranges column from the given start/end columns and a range function.
pub(super) fn temporal_ranges_impl_broadcast<T, U, F>(
    start: &ChunkedArray<T>,
    end: &ChunkedArray<T>,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<Series>
where
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
{
    match (start.len(), end.len()) {
        (len_start, len_end) if len_start == len_end => {
            build_temporal_ranges::<_, _, T, U, F>(
                start.downcast_iter().flatten(),
                end.downcast_iter().flatten(),
                range_impl,
                builder,
            )?;
        },
        (1, len_end) => {
            let start_scalar = start.get(0);
            match start_scalar {
                Some(start) => build_temporal_ranges::<_, _, T, U, F>(
                    std::iter::repeat(Some(&start)),
                    end.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_end),
            }
        },
        (len_start, 1) => {
            let end_scalar = end.get(0);
            match end_scalar {
                Some(end) => build_temporal_ranges::<_, _, T, U, F>(
                    start.downcast_iter().flatten(),
                    std::iter::repeat(Some(&end)),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_start),
            }
        },
        (len_start, len_end) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}) and `end` ({}) do not match",
                len_start, len_end
            )
        },
    };
    let out = builder.finish().into_series();
    Ok(out)
}

/// Iterate over a start and end column and create a range with the step for each entry.
fn build_numeric_ranges<'a, I, J, K, T, U, F>(
    start: I,
    end: J,
    step: K,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<()>
where
    I: Iterator<Item = Option<&'a T::Native>>,
    J: Iterator<Item = Option<&'a T::Native>>,
    K: Iterator<Item = Option<&'a i64>>,
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, i64, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
{
    for ((start, end), step) in start.zip(end).zip(step) {
        match (start, end, step) {
            (Some(start), Some(end), Some(step)) => range_impl(*start, *end, *step, builder)?,
            _ => builder.append_null(),
        }
    }
    Ok(())
}

/// Iterate over a start and end column and create a range for each entry.
fn build_temporal_ranges<'a, I, J, T, U, F>(
    start: I,
    end: J,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<()>
where
    I: Iterator<Item = Option<&'a T::Native>>,
    J: Iterator<Item = Option<&'a T::Native>>,
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
{
    for (start, end) in start.zip(end) {
        match (start, end) {
            (Some(start), Some(end)) => range_impl(*start, *end, builder)?,
            _ => builder.append_null(),
        }
    }
    Ok(())
}

/// Add `n` nulls to the builder.
fn build_nulls<U>(builder: &mut ListPrimitiveChunkedBuilder<U>, n: usize)
where
    U: PolarsIntegerType,
{
    for _ in 0..n {
        builder.append_null()
    }
}
