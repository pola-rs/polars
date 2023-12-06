use std::iter::zip;

use polars_core::prelude::*;
use polars_core::series::{IsSorted, Series};
use polars_core::with_match_physical_integer_polars_type;

use super::utils::{ensure_range_bounds_contain_exactly_one_value, ranges_impl_broadcast};

pub(super) fn int_range(s: &[Series], step: i64, dtype: DataType) -> PolarsResult<Series> {
    let mut start = &s[0];
    let mut end = &s[1];

    ensure_range_bounds_contain_exactly_one_value(start, end)?;
    polars_ensure!(dtype.is_integer(), ComputeError: "non-integer `dtype` passed to `int_range`: {:?}", dtype);

    let (start_storage, end_storage);
    if *start.dtype() != dtype {
        start_storage = start.strict_cast(&dtype)?;
        start = &start_storage;
    }
    if *end.dtype() != dtype {
        end_storage = end.strict_cast(&dtype)?;
        end = &end_storage;
    }

    with_match_physical_integer_polars_type!(dtype, |$T| {
        let start_v = get_first_series_value::<$T>(start)?;
        let end_v = get_first_series_value::<$T>(end)?;
        int_range_impl::<$T>(start_v, end_v, step)
    })
}

fn get_first_series_value<T>(s: &Series) -> PolarsResult<T::Native>
where
    T: PolarsIntegerType,
{
    let ca: &ChunkedArray<T> = s.as_any().downcast_ref().unwrap();
    let value_opt = ca.get(0);
    let value = value_opt.ok_or(polars_err!(ComputeError: "invalid null input for `int_range`"))?;
    Ok(value)
}

fn int_range_impl<T>(start: T::Native, end: T::Native, step: i64) -> PolarsResult<Series>
where
    T: PolarsIntegerType,
    ChunkedArray<T>: IntoSeries,
    std::ops::Range<T::Native>: DoubleEndedIterator<Item = T::Native>,
{
    let name = "int";

    let mut ca = match step {
        0 => polars_bail!(InvalidOperation: "step must not be zero"),
        1 => ChunkedArray::<T>::from_iter_values(name, start..end),
        2.. => ChunkedArray::<T>::from_iter_values(name, (start..end).step_by(step as usize)),
        _ => ChunkedArray::<T>::from_iter_values(
            name,
            (end..start)
                .step_by(step.unsigned_abs() as usize)
                .map(|x| start - (x - end)),
        ),
    };

    let is_sorted = if end < start {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    ca.set_sorted_flag(is_sorted);

    Ok(ca.into_series())
}

pub(super) fn int_ranges(s: &[Series], step: i64) -> PolarsResult<Series> {
    let start = &s[0].rechunk();
    let end = &s[1].rechunk();

    let mut start = start.cast(&DataType::Int64)?;
    let mut end = end.cast(&DataType::Int64)?;

    (start, end) = broadcast_scalar_inputs(start, end)?;

    let start = start.i64()?;
    let end = end.i64()?;

    // First do a pass to determine the required value capacity.
    let mut capacity = 0;
    let mut increment_capacity = |start: i64, end: i64| {
        if step == 1 {
            capacity += (end - start).unsigned_abs() as usize;
        } else {
            capacity += (((end - start).unsigned_abs() / step.unsigned_abs()) + 1) as usize;
        }
    };
    for (opt_start, opt_end) in zip(start, end) {
        if let (Some(start_v), Some(end_v)) = (opt_start, opt_end) {
            increment_capacity(start_v, end_v)
        }
    }

    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        "int_range",
        start.len(),
        capacity,
        DataType::Int64,
    );

    let range_impl = |start, end, builder: &mut ListPrimitiveChunkedBuilder<Int64Type>| {
        match step {
            1 => builder.append_iter_values(start..end),
            2.. => builder.append_iter_values((start..end).step_by(step as usize)),
            _ => builder.append_iter_values(
                (end..start)
                    .step_by(step.unsigned_abs() as usize)
                    .map(|x| start - (x - end)),
            ),
        };
        Ok(())
    };

    ranges_impl_broadcast(start, end, range_impl, &mut builder)
}

fn broadcast_scalar_inputs(start: Series, end: Series) -> PolarsResult<(Series, Series)> {
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
