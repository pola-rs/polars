use arrow::array::{Array, FixedSizeListArray, ListArray, StructArray};
use arrow::datatypes::ArrowDataType;
use arrow::types::Offset;
use polars_utils::IdxSize;
use polars_utils::itertools::Itertools;

use crate::cast::CastOptionsImpl;

/// Find the indices of the values where the validity mismatches.
///
/// This is done recursively, meaning that a validity mismatch at a deeper level will result as at
/// the level above at the corresponding index.
///
/// This procedure requires that
/// - Nulls are propagated recursively
/// - Lists to be
///     - trimmed to normalized offsets
///     - have the same number of child elements below each element (even nulls)
pub fn find_validity_mismatch(left: &dyn Array, right: &dyn Array, idxs: &mut Vec<IdxSize>) {
    assert_eq!(left.len(), right.len());

    // Handle the top-level.
    //
    // NOTE: This is done always, even if left and right have different nestings. This is
    // intentional and needed.
    let original_idxs_length = idxs.len();
    match (left.validity(), right.validity()) {
        (None, None) => {},
        (Some(l), Some(r)) => {
            if l != r {
                let mismatches = arrow::bitmap::xor(l, r);
                idxs.extend(mismatches.true_idx_iter().map(|i| i as IdxSize));
            }
        },
        (Some(v), _) | (_, Some(v)) => {
            if v.unset_bits() > 0 {
                let mismatches = !v;
                idxs.extend(mismatches.true_idx_iter().map(|i| i as IdxSize));
            }
        },
    }

    let left = left.as_any();
    let right = right.as_any();

    let pre_nesting_length = idxs.len();
    // (Struct, Struct)
    if let (Some(left), Some(right)) = (
        left.downcast_ref::<StructArray>(),
        right.downcast_ref::<StructArray>(),
    ) {
        assert_eq!(left.fields().len(), right.fields().len());
        for (l, r) in left.values().iter().zip(right.values().iter()) {
            find_validity_mismatch(l.as_ref(), r.as_ref(), idxs);
        }
    }

    // (List, List)
    if let (Some(left), Some(right)) = (
        left.downcast_ref::<ListArray<i32>>(),
        right.downcast_ref::<ListArray<i32>>(),
    ) {
        find_validity_mismatch_list_list_nested(left, right, idxs);
    }
    if let (Some(left), Some(right)) = (
        left.downcast_ref::<ListArray<i64>>(),
        right.downcast_ref::<ListArray<i64>>(),
    ) {
        find_validity_mismatch_list_list_nested(left, right, idxs);
    }

    // (FixedSizeList, FixedSizeList)
    if let (Some(left), Some(right)) = (
        left.downcast_ref::<FixedSizeListArray>(),
        right.downcast_ref::<FixedSizeListArray>(),
    ) {
        assert_eq!(left.size(), right.size());
        find_validity_mismatch_fsl_fsl_nested(
            left.values().as_ref(),
            right.values().as_ref(),
            left.size(),
            idxs,
        )
    }

    // (List, Array) / (Array, List)
    if let (Some(left), Some(right)) = (
        left.downcast_ref::<ListArray<i32>>(),
        right.downcast_ref::<FixedSizeListArray>(),
    ) {
        find_validity_mismatch_list_fsl_impl(left, right, idxs);
    }
    if let (Some(left), Some(right)) = (
        left.downcast_ref::<ListArray<i64>>(),
        right.downcast_ref::<FixedSizeListArray>(),
    ) {
        find_validity_mismatch_list_fsl_impl(left, right, idxs);
    }
    if let (Some(right), Some(left)) = (
        left.downcast_ref::<FixedSizeListArray>(),
        right.downcast_ref::<ListArray<i32>>(),
    ) {
        find_validity_mismatch_list_fsl_impl(left, right, idxs);
    }
    if let (Some(right), Some(left)) = (
        left.downcast_ref::<FixedSizeListArray>(),
        right.downcast_ref::<ListArray<i64>>(),
    ) {
        find_validity_mismatch_list_fsl_impl(left, right, idxs);
    }

    if pre_nesting_length == idxs.len() {
        return;
    }
    idxs[original_idxs_length..].sort_unstable();
}

fn find_validity_mismatch_fsl_fsl_nested(
    left: &dyn Array,
    right: &dyn Array,
    size: usize,
    idxs: &mut Vec<IdxSize>,
) {
    assert_eq!(left.len(), right.len());
    let start_length = idxs.len();
    find_validity_mismatch(left, right, idxs);
    if idxs.len() > start_length {
        let mut offset = 0;
        idxs[start_length] /= size as IdxSize;
        for i in start_length + 1..idxs.len() {
            idxs[i - offset] = idxs[i] / size as IdxSize;

            if idxs[i - offset] == idxs[i - offset - 1] {
                offset += 1;
            }
        }
        idxs.truncate(idxs.len() - offset);
    }
}

fn find_validity_mismatch_list_list_nested<O: Offset>(
    left: &ListArray<O>,
    right: &ListArray<O>,
    idxs: &mut Vec<IdxSize>,
) {
    let mut nested_idxs = Vec::new();
    find_validity_mismatch(
        left.values().as_ref(),
        right.values().as_ref(),
        &mut nested_idxs,
    );

    if nested_idxs.is_empty() {
        return;
    }

    assert_eq!(left.offsets().first().to_usize(), 0);
    assert_eq!(left.offsets().range().to_usize(), left.values().len());

    // @TODO: Optimize. This is only used on the error path so it is find, right?
    let mut j = 0;
    for (i, (start, length)) in left.offsets().offset_and_length_iter().enumerate_idx() {
        if j < nested_idxs.len() && (nested_idxs[j] as usize) < start + length {
            idxs.push(i);
            j += 1;

            // Loop over remaining items in same element.
            while j < nested_idxs.len() && (nested_idxs[j] as usize) < start + length {
                j += 1;
            }
        }

        if j == nested_idxs.len() {
            break;
        }
    }
}

fn find_validity_mismatch_list_fsl_impl<O: Offset>(
    left: &ListArray<O>,
    right: &FixedSizeListArray,
    idxs: &mut Vec<IdxSize>,
) {
    if left.validity().is_none() && right.validity().is_none() {
        find_validity_mismatch_fsl_fsl_nested(
            left.values().as_ref(),
            right.values().as_ref(),
            right.size(),
            idxs,
        );
        return;
    }

    let (ArrowDataType::List(f) | ArrowDataType::LargeList(f)) = left.dtype() else {
        unreachable!();
    };
    let left = crate::cast::cast_list_to_fixed_size_list(
        left,
        f,
        right.size(),
        CastOptionsImpl::default(),
    )
    .unwrap();
    find_validity_mismatch_fsl_fsl_nested(
        left.values().as_ref(),
        right.values().as_ref(),
        right.size(),
        idxs,
    )
}
