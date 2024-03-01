use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
use polars_ops::series::new_int_range;

use super::utils::{ensure_range_bounds_contain_exactly_one_value, numeric_ranges_impl_broadcast};

const CAPACITY_FACTOR: usize = 5;

pub(super) fn int_range(s: &[Series], step: i64, dtype: DataType) -> PolarsResult<Series> {
    let mut start = &s[0];
    let mut end = &s[1];
    let name = start.name();

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
        new_int_range::<$T>(start_v, end_v, step, name)
    })
}

fn get_first_series_value<T>(s: &Series) -> PolarsResult<T::Native>
where
    T: PolarsIntegerType,
{
    let ca: &ChunkedArray<T> = s.as_any().downcast_ref().unwrap();
    let value_opt = ca.get(0);
    let value =
        value_opt.ok_or_else(|| polars_err!(ComputeError: "invalid null input for `int_range`"))?;
    Ok(value)
}

pub(super) fn int_ranges(s: &[Series]) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];
    let step = &s[2];

    let start = start.cast(&DataType::Int64)?;
    let end = end.cast(&DataType::Int64)?;
    let step = step.cast(&DataType::Int64)?;

    let start = start.i64()?;
    let end = end.i64()?;
    let step = step.i64()?;

    let len = std::cmp::max(start.len(), end.len());
    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        // The name should follow our left hand rule.
        start.name(),
        len,
        len * CAPACITY_FACTOR,
        DataType::Int64,
    );

    let range_impl =
        |start, end, step: i64, builder: &mut ListPrimitiveChunkedBuilder<Int64Type>| {
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

    numeric_ranges_impl_broadcast(start, end, step, range_impl, &mut builder)
}
