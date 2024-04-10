use crate::prelude::*;
use crate::series::unstable::UnstableSeries;
use crate::series::IsSorted;

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

pub fn handle_casting_failures(input: &Series, output: &Series) -> PolarsResult<()> {
    let failure_mask = !input.is_null() & output.is_null();
    let failures = input.filter_threaded(&failure_mask, false)?;

    let additional_info = match (input.dtype(), output.dtype()) {
        (DataType::String, DataType::Date | DataType::Datetime(_, _)) => {
            "\n\nYou might want to try:\n\
            - setting `strict=False` to set values that cannot be converted to `null`\n\
            - using `str.strptime`, `str.to_date`, or `str.to_datetime` and providing a format string"
        },
        #[cfg(feature = "dtype-categorical")]
        (DataType::String, DataType::Enum(_,_)) => {
            "\n\nEnsure that all values in the input column are present in the categories of the enum datatype."
        }
        _ => "",
    };

    polars_bail!(
        ComputeError:
        "conversion from `{}` to `{}` failed in column '{}' for {} out of {} values: {}{}",
        input.dtype(),
        output.dtype(),
        output.name(),
        failures.len(),
        input.len(),
        failures.fmt_list(),
        additional_info,
    )
}
