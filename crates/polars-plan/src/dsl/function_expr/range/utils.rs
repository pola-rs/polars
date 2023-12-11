use std::iter::{repeat, zip};

use polars_core::prelude::{
    polars_bail, polars_ensure, ChunkedArray, IntoSeries, ListBuilderTrait,
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

/// Create a ranges column from the given start/end columns and a range function.
pub(super) fn ranges_impl_broadcast<T, U, F>(
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
            build_ranges::<T, U, F>(start.into_iter(), end.into_iter(), range_impl, builder)?;
        },
        (1, len_end) => {
            let start_scalar = start.get(0);
            match start_scalar {
                Some(start) => build_ranges::<T, U, F>(
                    repeat(Some(start)),
                    end.into_iter(),
                    range_impl,
                    builder,
                )?,
                None => build_nulls(builder, len_end),
            }
        },
        (len_start, 1) => {
            let end_scalar = end.get(0);
            match end_scalar {
                Some(end) => build_ranges::<T, U, F>(
                    start.into_iter(),
                    repeat(Some(end)),
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

/// Iterate over a start and end column and create a range for each entry.
fn build_ranges<T, U, F>(
    start: impl Iterator<Item = Option<T::Native>>,
    end: impl Iterator<Item = Option<T::Native>>,
    range_impl: F,
    builder: &mut ListPrimitiveChunkedBuilder<U>,
) -> PolarsResult<()>
where
    T: PolarsIntegerType,
    U: PolarsIntegerType,
    F: Fn(T::Native, T::Native, &mut ListPrimitiveChunkedBuilder<U>) -> PolarsResult<()>,
{
    for (start, end) in zip(start, end) {
        match (start, end) {
            (Some(start), Some(end)) => range_impl(start, end, builder)?,
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
