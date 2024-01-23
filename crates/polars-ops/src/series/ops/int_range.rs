use polars_core::prelude::*;
use polars_core::series::IsSorted;

pub fn new_int_range<T>(
    start: T::Native,
    end: T::Native,
    step: i64,
    name: &str,
) -> PolarsResult<Series>
where
    T: PolarsIntegerType,
    ChunkedArray<T>: IntoSeries,
    std::ops::Range<T::Native>: DoubleEndedIterator<Item = T::Native>,
{
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
