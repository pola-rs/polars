use std::rc::Rc;

use crate::prelude::*;
use crate::series::amortized_iter::AmortSeries;

/// A utility that allocates an [`AmortSeries`]. The applied function can then use that
/// series container to save heap allocations and swap arrow arrays.
pub fn with_unstable_series<F, T>(dtype: &DataType, f: F) -> T
where
    F: Fn(&mut AmortSeries) -> T,
{
    let container = Series::full_null("", 0, dtype);
    let mut us = AmortSeries::new(Rc::new(container));

    f(&mut us)
}

pub fn handle_casting_failures(input: &Series, output: &Series) -> PolarsResult<()> {
    let failure_mask = !input.is_null() & output.is_null();
    let failures = input.filter(&failure_mask)?;

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
        InvalidOperation:
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
