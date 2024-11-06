use polars_error::{polars_bail, PolarsError, PolarsResult};

use crate::array::growable::make_growable;
use crate::array::{Array, ArrayRef, FixedSizeListArray, PrimitiveArray};
use crate::bitmap::BitmapBuilder;
use crate::compute::utils::combine_validities_and;
use crate::datatypes::ArrowDataType;

pub fn sub_fixed_size_list_get_literal(
    arr: &FixedSizeListArray,
    index: i64,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    if cfg!(debug_assertions) {
        let msg = "fn sub_fixed_size_list_get_literal";
        dbg!(msg);
    }

    let ArrowDataType::FixedSizeList(_, width) = arr.dtype() else {
        unreachable!();
    };

    let width = *width;

    let index = usize::try_from(index).unwrap();

    if !null_on_oob && index >= width {
        polars_bail!(
            ComputeError:
            "get index {} is out of bounds for array(width={})",
            index,
            width
        );
    }

    let values = arr.values();

    let mut growable = make_growable(&[values.as_ref()], values.validity().is_some(), arr.len());

    for i in 0..arr.len() {
        unsafe { growable.extend(0, i * width + index, 1) }
    }

    Ok(growable.as_box())
}

pub fn sub_fixed_size_list_get(
    arr: &FixedSizeListArray,
    index: &PrimitiveArray<i64>,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    if cfg!(debug_assertions) {
        let msg = "fn sub_fixed_size_list_get";
        dbg!(msg);
    }

    fn idx_oob_err(index: i64, width: usize) -> PolarsError {
        PolarsError::ComputeError(
            format!(
                "get index {} is out of bounds for array(width={})",
                index, width
            )
            .into(),
        )
    }

    let ArrowDataType::FixedSizeList(_, width) = arr.dtype() else {
        unreachable!();
    };

    let width = *width;

    if arr.is_empty() {
        if !null_on_oob {
            if let Some(i) = index.non_null_values_iter().max() {
                if usize::try_from(i).unwrap() >= width {
                    return Err(idx_oob_err(i, width));
                }
            }
        }

        let values = arr.values();
        assert!(values.is_empty());
        return Ok(values.clone());
    }

    if !null_on_oob && width == 0 {
        if let Some(i) = index.non_null_values_iter().next() {
            return Err(idx_oob_err(i, width));
        }
    }

    // Array is non-empty and has non-zero width at this point
    let values = arr.values();

    let mut growable = make_growable(&[values.as_ref()], values.validity().is_some(), arr.len());
    let mut output_validity = BitmapBuilder::with_capacity(arr.len());
    let opt_index_validity = index.validity();
    let mut exceeded_width_idx = 0;

    for i in 0..arr.len() {
        let idx = usize::try_from(index.value(i)).unwrap();
        let idx_is_oob = idx >= width;
        let idx_is_valid = opt_index_validity.map_or(true, |x| unsafe { x.get_bit_unchecked(i) });

        if idx_is_oob && idx_is_valid && exceeded_width_idx < width {
            exceeded_width_idx = idx;
        }

        let idx = if idx_is_oob { 0 } else { idx };

        unsafe {
            growable.extend(0, i * width + idx, 1);
            let output_is_valid = idx_is_valid & !idx_is_oob;
            output_validity.push_unchecked(output_is_valid);
        }
    }

    if !null_on_oob && exceeded_width_idx >= width {
        return Err(idx_oob_err(exceeded_width_idx as i64, width));
    }

    let output = growable.as_box();
    let output_validity = combine_validities_and(Some(&output_validity.freeze()), arr.validity());

    Ok(output.with_validity(output_validity))
}
