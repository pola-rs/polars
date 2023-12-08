use polars_core::prelude::*;
use polars_core::series::{IsSorted, Series};
use polars_core::with_match_physical_integer_polars_type;

use super::utils::ensure_range_bounds_contain_exactly_one_value;

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
