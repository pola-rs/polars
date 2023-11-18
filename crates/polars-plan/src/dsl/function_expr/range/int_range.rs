use polars_core::prelude::*;
use polars_core::series::{IsSorted, Series};

use super::utils::ensure_range_bounds_contain_exactly_one_value;

pub(super) fn int_range(s: &[Series], step: i64) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    ensure_range_bounds_contain_exactly_one_value(start, end)?;

    match start.dtype() {
        dt if dt == &IDX_DTYPE => {
            let start = start.idx()?.get(0).unwrap();
            let end = end.cast(&IDX_DTYPE)?;
            let end = end.idx()?.get(0).unwrap();
            int_range_impl::<IdxType>(start, end, step)
        },
        _ => {
            let start = start.cast(&DataType::Int64)?;
            let end = end.cast(&DataType::Int64)?;
            let start = start.i64()?.get(0).unwrap();
            let end = end.i64()?.get(0).unwrap();
            int_range_impl::<Int64Type>(start, end, step)
        },
    }
}

pub(super) fn int_ranges(s: &[Series], step: i64) -> PolarsResult<Series> {
    let start = &s[0].rechunk();
    let end = &s[1].rechunk();

    let output_name = "int_range";

    let mut start = start.cast(&DataType::Int64)?;
    let mut end = end.cast(&DataType::Int64)?;

    if start.len() != end.len() {
        if start.len() == 1 {
            start = start.new_from_index(0, end.len())
        } else if end.len() == 1 {
            end = end.new_from_index(0, start.len())
        } else {
            polars_bail!(
                ComputeError:
                "lengths of `start`: {} and `end`: {} arguments `\
                cannot be matched in the `int_ranges` expression",
                start.len(), end.len()
            );
        }
    }

    let start = start.i64()?;
    let end = end.i64()?;

    let start = start.downcast_iter().next().unwrap();
    let end = end.downcast_iter().next().unwrap();

    // First do a pass to determine the required value capacity.
    let mut values_capacity = 0;
    for (opt_start, opt_end) in start.into_iter().zip(end) {
        if let (Some(start_v), Some(end_v)) = (opt_start, opt_end) {
            if step == 1 {
                values_capacity += (end_v - start_v).unsigned_abs() as usize;
            } else {
                values_capacity +=
                    (((end_v - start_v).unsigned_abs() / step.unsigned_abs()) + 1) as usize;
            }
        }
    }

    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        output_name,
        start.len(),
        values_capacity,
        DataType::Int64,
    );

    for (opt_start, opt_end) in start.into_iter().zip(end) {
        match (opt_start, opt_end) {
            (Some(&start_v), Some(&end_v)) => match step {
                1 => {
                    builder.append_iter_values(start_v..end_v);
                },
                2.. => {
                    builder.append_iter_values((start_v..end_v).step_by(step as usize));
                },
                _ => builder.append_iter_values(
                    (end_v..start_v)
                        .step_by(step.unsigned_abs() as usize)
                        .map(|x| start_v - (x - end_v)),
                ),
            },
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

fn int_range_impl<T>(start: T::Native, end: T::Native, step: i64) -> PolarsResult<Series>
where
    T: PolarsNumericType,
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
