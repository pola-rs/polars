use std::rc::Rc;

use crate::prelude::*;
use crate::series::amortized_iter::AmortSeries;

/// A utility that allocates an [`AmortSeries`]. The applied function can then use that
/// series container to save heap allocations and swap arrow arrays.
pub fn with_unstable_series<F, T>(dtype: &DataType, f: F) -> T
where
    F: Fn(&mut AmortSeries) -> T,
{
    let container = Series::full_null(PlSmallStr::EMPTY, 0, dtype);
    let mut us = AmortSeries::new(Rc::new(container));

    f(&mut us)
}

pub fn handle_casting_failures(input: &Series, output: &Series) -> PolarsResult<()> {
    use DataType as D;
    #[allow(clippy::single_match)]
    match (input.dtype(), output.dtype()) {
        // @Hack to deal with deprecated cast
        // @2.0
        #[cfg(feature = "dtype-struct")]
        (D::Struct(l_fields), D::Struct(r_fields)) => {
            if l_fields.len() != r_fields.len()
                || l_fields
                    .iter()
                    .zip(r_fields.iter())
                    .any(|(l, r)| l.name() != r.name())
            {
                return Ok(());
            }
        },
        _ => {},
    }

    let mut idxs = Vec::new();
    input.find_validity_mismatch(output, &mut idxs);

    // Base case. No strict casting failed.
    if idxs.is_empty() {
        return Ok(());
    }

    let num_failures = idxs.len();
    let failures = input.take_slice(&idxs[..num_failures.min(10)])?;

    let additional_info = match (input.dtype(), output.dtype()) {
        (DataType::String, DataType::Date | DataType::Datetime(_, _)) => {
            "\n\nYou might want to try:\n\
            - setting `strict=False` to set values that cannot be converted to `null`\n\
            - using `str.strptime`, `str.to_date`, or `str.to_datetime` and providing a format string"
        },
        #[cfg(feature = "dtype-categorical")]
        (DataType::String, DataType::Enum(_, _)) => {
            "\n\nEnsure that all values in the input column are present in the categories of the enum datatype."
        },
        _ if failures.len() < num_failures => {
            "\n\nDid not show all failed cases as there were too many."
        },
        _ => "",
    };

    polars_bail!(
        InvalidOperation:
        "conversion from `{}` to `{}` failed in column '{}' for {} out of {} values: {}{}",
        input.dtype(),
        output.dtype(),
        output.name(),
        num_failures,
        input.len(),
        failures.fmt_list(),
        additional_info,
    )
}
