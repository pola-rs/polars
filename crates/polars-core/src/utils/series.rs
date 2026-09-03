use std::rc::Rc;

use polars_compute::find_validity_mismatch::find_validity_mismatch;
use polars_compute::gather::take_unchecked;

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

pub fn check_is_valid_struct_cast(
    input_dtype: &DataType,
    output_dtype: &DataType,
    output_name: &PlSmallStr,
) -> PolarsResult<()> {
    use DataType as D;

    let err = |msg: &str| -> PolarsError {
        polars_err!(
            InvalidOperation:
            "cast from `{}` to `{}` failed in column '{}': {}\n\n\
            Ensure that any output struct has the same number of fields as the input, and that all struct field names in the output are present in the input.\n\
            Use `strict=False` to force the cast, and Polars will select the first n fields from the struct.",
            input_dtype,
            output_dtype,
            output_name,
            msg,
        )
    };

    #[allow(clippy::single_match)]
    match (input_dtype, output_dtype) {
        #[cfg(feature = "dtype-struct")]
        (D::Struct(l_fields), D::Struct(r_fields)) => {
            if l_fields.len() != r_fields.len() {
                return Err(err(&format!(
                    "structs do not have the same number of fields: {} vs {}",
                    l_fields.len(),
                    r_fields.len(),
                )));
            }
            for (l, r) in Iterator::zip(l_fields.iter(), r_fields.iter()) {
                if l.name() != r.name() {
                    return Err(err(&format!(
                        "structs field name mismatch: {} vs {}",
                        l.name(),
                        r.name()
                    )));
                }
                check_is_valid_struct_cast(l.dtype(), r.dtype(), output_name)?;
            }
            Ok(())
        },
        (D::List(input_dtype), D::List(output_dtype)) => {
            check_is_valid_struct_cast(input_dtype, output_dtype, output_name)
        },
        #[cfg(feature = "dtype-array")]
        (D::Array(input_dtype, _), D::Array(output_dtype, _)) => {
            check_is_valid_struct_cast(input_dtype, output_dtype, output_name)
        },
        #[cfg(feature = "dtype-array")]
        (D::List(input_dtype), D::Array(output_dtype, _))
        | (D::Array(input_dtype, _), D::List(output_dtype)) => {
            check_is_valid_struct_cast(input_dtype, output_dtype, output_name)
        },
        _ => Ok(()),
    }
}

pub fn handle_casting_failures(input: &Series, output: &Series) -> PolarsResult<()> {
    check_is_valid_struct_cast(input.dtype(), output.dtype(), output.name())?;

    let mut idxs = Vec::new();
    input.find_validity_mismatch(output, &mut idxs);

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

pub fn handle_array_casting_failures(input: &dyn Array, output: &dyn Array) -> PolarsResult<()> {
    let mut idxs = Vec::new();
    find_validity_mismatch(input, output, &mut idxs);
    if idxs.is_empty() {
        return Ok(());
    }

    let num_failures = idxs.len();
    let failures = PrimitiveArray::with_slice(&idxs[..num_failures.min(10)], |idxs| unsafe {
        take_unchecked(input, &idxs)
    });

    polars_bail!(
        InvalidOperation:
        "conversion from `{}` to `{}` failed for {} out of {} values: {}",
        DataType::from_arrow(input.dtype(), None),
        DataType::from_arrow(output.dtype(), None),
        num_failures,
        input.len(),
        Series::try_from((PlSmallStr::EMPTY, failures))?,
    )
}
