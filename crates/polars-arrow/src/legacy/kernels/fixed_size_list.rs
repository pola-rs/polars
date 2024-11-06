use polars_error::{polars_bail, PolarsError, PolarsResult};

use crate::array::growable::make_growable;
use crate::array::{Array, ArrayRef, FixedSizeListArray, PrimitiveArray};
use crate::bitmap::BitmapBuilder;
use crate::compute::utils::combine_validities_and3;
use crate::datatypes::ArrowDataType;

pub fn sub_fixed_size_list_get_literal(
    arr: &FixedSizeListArray,
    index: i64,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    let ArrowDataType::FixedSizeList(_, width) = arr.dtype() else {
        unreachable!();
    };

    let width = *width;

    let orig_index = index;

    let index = if index < 0 {
        if index.unsigned_abs() as usize > width {
            width
        } else {
            (width as i64 + index) as usize
        }
    } else {
        usize::try_from(index).unwrap()
    };

    if !null_on_oob && index >= width {
        polars_bail!(
            ComputeError:
            "get index {} is out of bounds for array(width={})",
            orig_index,
            width
        );
    }

    let values = arr.values();

    let mut growable = make_growable(&[values.as_ref()], values.validity().is_some(), arr.len());

    if index >= width {
        unsafe { growable.extend_validity(arr.len()) }
        return Ok(growable.as_box());
    }

    if let Some(arr_validity) = arr.validity() {
        for i in 0..arr.len() {
            unsafe {
                if arr_validity.get_bit_unchecked(i) {
                    growable.extend(0, i * width + index, 1)
                } else {
                    growable.extend_validity(1)
                }
            }
        }
    } else {
        for i in 0..arr.len() {
            unsafe { growable.extend(0, i * width + index, 1) }
        }
    }

    Ok(growable.as_box())
}

pub fn sub_fixed_size_list_get(
    arr: &FixedSizeListArray,
    index: &PrimitiveArray<i64>,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    assert_eq!(arr.len(), index.len());

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
    let mut idx_oob_validity = BitmapBuilder::with_capacity(arr.len());
    let opt_index_validity = index.validity();
    let mut exceeded_width_idx = 0;
    let mut current_index_i64 = 0;

    for i in 0..arr.len() {
        let index = index.value(i);
        current_index_i64 = index;

        let idx = if index < 0 {
            if index.unsigned_abs() as usize > width {
                width
            } else {
                (width as i64 + index) as usize
            }
        } else {
            usize::try_from(index).unwrap()
        };

        let idx_is_oob = idx >= width;
        let idx_is_valid = opt_index_validity.map_or(true, |x| unsafe { x.get_bit_unchecked(i) });

        if idx_is_oob && idx_is_valid && exceeded_width_idx < width {
            exceeded_width_idx = idx;
        }

        let idx = if idx_is_oob { 0 } else { idx };

        unsafe {
            growable.extend(0, i * width + idx, 1);
            let output_is_valid = idx_is_valid & !idx_is_oob;
            idx_oob_validity.push_unchecked(output_is_valid);
        }
    }

    if !null_on_oob && exceeded_width_idx >= width {
        return Err(idx_oob_err(current_index_i64, width));
    }

    let output = growable.as_box();
    let output_validity = combine_validities_and3(
        output.validity(),                // inner validity
        Some(&idx_oob_validity.freeze()), // validity for OOB idx
        arr.validity(),                   // outer validity
    );

    Ok(output.with_validity(output_validity))
}
