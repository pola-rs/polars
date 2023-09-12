use crate::prelude::*;
use crate::series::unstable::UnstableSeries;
use crate::series::IsSorted;

/// Transform to physical type and coerce floating point and similar sized integer to a bit representation
/// to reduce compiler bloat
pub fn _to_physical_and_bit_repr(s: &[Series]) -> Vec<Series> {
    s.iter()
        .map(|s| {
            let physical = s.to_physical_repr();
            match physical.dtype() {
                DataType::Int64 => physical.bit_repr_large().into_series(),
                DataType::Int32 => physical.bit_repr_small().into_series(),
                DataType::Float32 => physical.bit_repr_small().into_series(),
                DataType::Float64 => physical.bit_repr_large().into_series(),
                _ => physical.into_owned(),
            }
        })
        .collect()
}

/// A utility that allocates an [`UnstableSeries`]. The applied function can then use that
/// series container to save heap allocations and swap arrow arrays.
pub fn with_unstable_series<F, T>(dtype: &DataType, f: F) -> T
where
    F: Fn(&mut UnstableSeries) -> T,
{
    let mut container = Series::full_null("", 0, dtype);
    let mut us = UnstableSeries::new(&mut container);

    f(&mut us)
}

pub fn ensure_sorted_arg(s: &Series, operation: &str) -> PolarsResult<()> {
    polars_ensure!(!matches!(s.is_sorted_flag(), IsSorted::Not), InvalidOperation: "argument in operation '{}' is not explicitly sorted

- If your data is ALREADY sorted, set the sorted flag with: '.set_sorted()'.
- If your data is NOT sorted, sort the 'expr/series/column' first.
    ", operation);
    Ok(())
}

pub fn get_casting_failures(input: &Series, output: &Series) -> PolarsResult<Series> {
    let failure_mask = !input.is_null() & output.is_null();
    input.filter_threaded(&failure_mask, false)?.unique()
}
