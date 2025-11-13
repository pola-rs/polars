use polars_core::prelude::{
    ChunkedArray, Column, Int64Chunked, IntoColumn, ListBuilderTrait, ListPrimitiveChunkedBuilder,
    PolarsIntegerType, PolarsNumericType, PolarsResult, polars_bail, polars_ensure,
};

pub(super) fn temporal_series_to_i64_scalar(s: &Column) -> Option<i64> {
    s.to_physical_repr().get(0).unwrap().extract::<i64>()
}
pub(super) fn ensure_items_contain_exactly_one_value(
    values: &[&Column],
    names: &[&str],
) -> PolarsResult<()> {
    for (value, name) in values.iter().zip(names.iter()) {
        polars_ensure!(
            value.len() == 1,
            ComputeError: "`{name}` must contain exactly one value, got {} values", value.len()
        )
    }
    Ok(())
}

/// Create a numeric ranges column from the given start/end/step columns and a range function.
pub(super) fn numeric_ranges_impl_broadcast<T, U, F>(
    start: &ChunkedArray<T>,
    end: &ChunkedArray<T>,
    step: &Int64Chunked,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<Column>
where
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, i64, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
    ListPrimitiveChunkedBuilder<U>: ListBuilderTrait,
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
    let out = builder.finish().into_column();
    Ok(out)
}

/// Create a ranges column from two columns and a range function.
pub(super) fn temporal_ranges_impl_broadcast_2args<T, U, F>(
    s1: &ChunkedArray<T>,
    s2: &ChunkedArray<T>,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<Column>
where
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
    ListPrimitiveChunkedBuilder<U>: ListBuilderTrait,
{
    match (s1.len(), s2.len()) {
        (len_s1, len_s2) if len_s1 == len_s2 => {
            build_temporal_ranges_2args::<_, _, T, U, F>(
                s1.downcast_iter().flatten(),
                s2.downcast_iter().flatten(),
                range_impl,
                builder,
            )?;
        },
        (1, len_s2) => {
            let s1_scalar = s1.get(0);
            match s1_scalar {
                Some(s1) => build_temporal_ranges_2args::<_, _, T, U, F>(
                    std::iter::repeat(Some(&s1)),
                    s2.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_s2),
            }
        },
        (len_s1, 1) => {
            let s2_scalar = s2.get(0);
            match s2_scalar {
                Some(s2) => build_temporal_ranges_2args::<_, _, T, U, F>(
                    s1.downcast_iter().flatten(),
                    std::iter::repeat(Some(&s2)),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_s1),
            }
        },
        (len_s1, len_s2) => {
            polars_bail!(
                ComputeError:
                "lengths of `s1` ({}) and `s2` ({}) do not match",
                len_s1, len_s2
            )
        },
    };
    let out = builder.finish().into_column();
    Ok(out)
}

/// Create a ranges column from two columns and a range function.
pub(super) fn temporal_ranges_impl_broadcast_3args<T, U, F>(
    s1: &ChunkedArray<T>,
    s2: &ChunkedArray<T>,
    s3: &ChunkedArray<T>,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<Column>
where
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, T::Native, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
    ListPrimitiveChunkedBuilder<U>: ListBuilderTrait,
{
    match (s1.len(), s2.len(), s3.len()) {
        (len1, len2, len3) if len1 == len2 && len1 == len3 => {
            build_temporal_ranges_3args::<_, _, _, T, U, F>(
                s1.downcast_iter().flatten(),
                s2.downcast_iter().flatten(),
                s3.downcast_iter().flatten(),
                range_impl,
                builder,
            )?;
        },
        (len1, len2, 1) if (len1 == len2) => {
            let s3_scalar = s3.get(0);
            match s3_scalar {
                Some(s3) => build_temporal_ranges_3args::<_, _, _, T, U, F>(
                    s1.downcast_iter().flatten(),
                    s2.downcast_iter().flatten(),
                    std::iter::repeat(Some(&s3)),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len1),
            }
        },
        (len1, 1, len3) if (len1 == len3) => {
            let s2_scalar = s2.get(0);
            match s2_scalar {
                Some(s2) => build_temporal_ranges_3args::<_, _, _, T, U, F>(
                    s1.downcast_iter().flatten(),
                    std::iter::repeat(Some(&s2)),
                    s3.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len1),
            }
        },
        (1, len2, len3) if (len2 == len3) => {
            let s1_scalar = s1.get(0);
            match s1_scalar {
                Some(s1) => build_temporal_ranges_3args::<_, _, _, T, U, F>(
                    std::iter::repeat(Some(&s1)),
                    s2.downcast_iter().flatten(),
                    s3.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len2),
            }
        },
        (1, 1, len3) => {
            let s1_scalar = s1.get(0);
            let s2_scalar = s2.get(0);
            match (s1_scalar, s2_scalar) {
                (Some(s1), Some(s2)) => build_temporal_ranges_3args::<_, _, _, T, U, F>(
                    std::iter::repeat(Some(&s1)),
                    std::iter::repeat(Some(&s2)),
                    s3.downcast_iter().flatten(),
                    range_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len3),
            }
        },
        (1, len2, 1) => {
            let s1_scalar = s1.get(0);
            let s3_scalar = s3.get(0);
            match (s1_scalar, s3_scalar) {
                (Some(s1), Some(s3)) => build_temporal_ranges_3args::<_, _, _, T, U, F>(
                    std::iter::repeat(Some(&s1)),
                    s2.downcast_iter().flatten(),
                    std::iter::repeat(Some(&s3)),
                    range_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len2),
            }
        },
        (len1, 1, 1) => {
            let s2_scalar = s2.get(0);
            let s3_scalar = s3.get(0);
            match (s2_scalar, s3_scalar) {
                (Some(s2), Some(s3)) => build_temporal_ranges_3args::<_, _, _, T, U, F>(
                    s1.downcast_iter().flatten(),
                    std::iter::repeat(Some(&s2)),
                    std::iter::repeat(Some(&s3)),
                    range_impl,
                    builder,
                )?,
                _ => build_nulls(builder, len1),
            }
        },
        (len1, len2, len3) => {
            polars_bail!(
                ComputeError:
                "lengths of `s1` ({}), `s2` ({}), and `s3` ({}) do not match",
                len1, len2, len3
            )
        },
    };
    let out = builder.finish().into_column();
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
    ListPrimitiveChunkedBuilder<U>: ListBuilderTrait,
{
    for ((start, end), step) in start.zip(end).zip(step) {
        match (start, end, step) {
            (Some(start), Some(end), Some(step)) => range_impl(*start, *end, *step, builder)?,
            _ => builder.append_null(),
        }
    }
    Ok(())
}

/// Iterate over two columns and create a range for each entry.
fn build_temporal_ranges_2args<'a, I, J, T, U, F>(
    s1: I,
    s2: J,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<()>
where
    I: Iterator<Item = Option<&'a T::Native>>,
    J: Iterator<Item = Option<&'a T::Native>>,
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
    ListPrimitiveChunkedBuilder<U>: ListBuilderTrait,
{
    for (s1, s2) in s1.zip(s2) {
        match (s1, s2) {
            (Some(s1), Some(s2)) => range_impl(*s1, *s2, builder)?,
            _ => builder.append_null(),
        }
    }
    Ok(())
}
/// Iterate over two columns and create a range for each entry.
fn build_temporal_ranges_3args<'a, I, J, K, T, U, F>(
    s1: I,
    s2: J,
    s3: K,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<()>
where
    I: Iterator<Item = Option<&'a T::Native>>,
    J: Iterator<Item = Option<&'a T::Native>>,
    K: Iterator<Item = Option<&'a T::Native>>,
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, T::Native, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
    ListPrimitiveChunkedBuilder<U>: ListBuilderTrait,
{
    for ((s1, s2), s3) in s1.zip(s2).zip(s3) {
        match (s1, s2, s3) {
            (Some(s1), Some(s2), Some(s3)) => range_impl(*s1, *s2, *s3, builder)?,
            _ => builder.append_null(),
        }
    }
    Ok(())
}

/// Add `n` nulls to the builder.
pub fn build_nulls<U>(builder: &mut ListPrimitiveChunkedBuilder<U>, n: usize)
where
    U: PolarsNumericType,
    ListPrimitiveChunkedBuilder<U>: ListBuilderTrait,
{
    for _ in 0..n {
        builder.append_null()
    }
}
