use std::iter::zip;

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

pub(super) fn broadcast_scalar_inputs(
    start: Series,
    end: Series,
) -> PolarsResult<(Series, Series)> {
    match (start.len(), end.len()) {
        (len1, len2) if len1 == len2 => Ok((start, end)),
        (1, len2) => {
            let start_matched = start.new_from_index(0, len2);
            Ok((start_matched, end))
        },
        (len1, 1) => {
            let end_matched = end.new_from_index(0, len1);
            Ok((start, end_matched))
        },
        (len1, len2) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}) and `end` ({}) do not match",
                len1, len2
            )
        },
    }
}

/// Create a ranges column from the given start/end columns and a range function.
pub(super) fn ranges_impl_broadcast<T, F>(
    builder: &mut ListPrimitiveChunkedBuilder<T>,
    start: &ChunkedArray<T>,
    end: &ChunkedArray<T>,
    range_impl: F,
) -> PolarsResult<Series>
where
    T: PolarsIntegerType,
    F: Fn(T::Native, T::Native) -> PolarsResult<ChunkedArray<T>>,
{
    match (start.len(), end.len()) {
        (len_start, len_end) if len_start == len_end => {
            ranges_double_impl(builder, start, end, range_impl)?;
        },
        (1, len_end) => {
            let start_scalar = unsafe { start.get_unchecked(0) };
            match start_scalar {
                Some(start) => {
                    let range_impl = |end| range_impl(start, end);
                    ranges_single_impl(builder, end, range_impl)?
                },
                None => build_nulls(builder, len_end),
            }
        },
        (len_start, 1) => {
            let end_scalar = unsafe { end.get_unchecked(0) };
            match end_scalar {
                Some(end) => {
                    let range_impl = |start| range_impl(start, end);
                    ranges_single_impl(builder, start, range_impl)?
                },
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

/// Iterate over a start AND end column and create a range for each entry.
fn ranges_double_impl<T, F>(
    builder: &mut ListPrimitiveChunkedBuilder<T>,
    start: &ChunkedArray<T>,
    end: &ChunkedArray<T>,
    range_impl: F,
) -> PolarsResult<()>
where
    T: PolarsIntegerType,
    F: Fn(T::Native, T::Native) -> PolarsResult<ChunkedArray<T>>,
{
    for (start, end) in zip(start, end) {
        match (start, end) {
            (Some(start), Some(end)) => {
                let rng = range_impl(start, end)?;
                builder.append_slice(rng.cont_slice().unwrap())
            },
            _ => builder.append_null(),
        }
    }
    Ok(())
}

/// Iterate over a start OR end column and create a range for each entry.
fn ranges_single_impl<T, F>(
    builder: &mut ListPrimitiveChunkedBuilder<T>,
    ca: &ChunkedArray<T>,
    range_impl: F,
) -> PolarsResult<()>
where
    T: PolarsIntegerType,
    F: Fn(T::Native) -> PolarsResult<ChunkedArray<T>>,
{
    for ca_scalar in ca {
        match ca_scalar {
            Some(ca_scalar) => {
                let rng = range_impl(ca_scalar)?;
                builder.append_slice(rng.cont_slice().unwrap())
            },
            None => builder.append_null(),
        }
    }
    Ok(())
}

/// Add nulls to the builder.
fn build_nulls<T>(builder: &mut ListPrimitiveChunkedBuilder<T>, n: usize)
where
    T: PolarsIntegerType,
{
    for _ in 0..n {
        builder.append_null()
    }
}
