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

pub fn sub_fixed_size_list_get_literal(arr: &FixedSizeListArray, index: i64) -> ArrayRef {
    let take_by = sub_fixed_size_list_get_indexes_literal(arr.size(), arr.len(), index);
    let values = arr.values();
    // Safety:
    // the indices we generate are in bounds
    unsafe { take_unchecked(&**values, &take_by) }
}

pub fn sub_fixed_size_list_get(arr: &FixedSizeListArray, index: &PrimitiveArray<i64>) -> ArrayRef {
    let take_by = sub_fixed_size_list_get_indexes(arr.size(), index);
    let values = arr.values();
    // Safety:
    // the indices we generate are in bounds
    unsafe { take_unchecked(&**values, &take_by) }
}
