use polars_error::{polars_bail, PolarsResult};
use polars_utils::index::NullCount;
use polars_utils::IdxSize;

use crate::array::{ArrayRef, FixedSizeListArray, PrimitiveArray};
use crate::compute::take::take_unchecked;
use crate::legacy::prelude::*;
use crate::legacy::utils::CustomIterTools;

fn sub_fixed_size_list_get_indexes_literal(width: usize, len: usize, index: i64) -> IdxArr {
    (0..len)
        .map(|i| {
            if index >= width as i64 {
                return None;
            }

            index
                .negative_to_usize(width)
                .map(|idx| (idx + i * width) as IdxSize)
        })
        .collect_trusted()
}

fn sub_fixed_size_list_get_indexes(width: usize, index: &PrimitiveArray<i64>) -> IdxArr {
    index
        .iter()
        .enumerate()
        .map(|(i, idx)| {
            if let Some(idx) = idx {
                if *idx >= width as i64 {
                    return None;
                }

                idx.negative_to_usize(width)
                    .map(|idx| (idx + i * width) as IdxSize)
            } else {
                None
            }
        })
        .collect_trusted()
}

pub fn sub_fixed_size_list_get_literal(
    arr: &FixedSizeListArray,
    index: i64,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    let take_by = sub_fixed_size_list_get_indexes_literal(arr.size(), arr.len(), index);
    if !null_on_oob && take_by.null_count() > 0 {
        polars_bail!(ComputeError: "get index is out of bounds");
    }

    let values = arr.values();
    // SAFETY:
    // the indices we generate are in bounds
    unsafe { Ok(take_unchecked(&**values, &take_by)) }
}

pub fn sub_fixed_size_list_get(
    arr: &FixedSizeListArray,
    index: &PrimitiveArray<i64>,
    null_on_oob: bool,
) -> PolarsResult<ArrayRef> {
    let take_by = sub_fixed_size_list_get_indexes(arr.size(), index);
    if !null_on_oob && take_by.null_count() > 0 {
        polars_bail!(ComputeError: "get index is out of bounds");
    }

    let values = arr.values();
    // SAFETY:
    // the indices we generate are in bounds
    unsafe { Ok(take_unchecked(&**values, &take_by)) }
}
