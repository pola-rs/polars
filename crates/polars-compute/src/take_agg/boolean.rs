#![allow(unsafe_op_in_unsafe_fn)]
//! Gather-and-reduce over a [`PlBooleanArray`].

use polars_array::PlBooleanArray;

/// The position in `indices` of the first index that gathers `extreme`, or any non-null value.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
unsafe fn take_arg_bool_nulls<I: IntoIterator<Item = usize>>(
    arr: &PlBooleanArray,
    indices: I,
    extreme: bool,
) -> Option<usize> {
    let validity = arr
        .validity()
        .expect("a chunk with nulls in it holds a validity mask");

    // Every element is null, so no index gathers anything.
    if validity.scalar_value() == Some(false) {
        return None;
    }

    // Every index gathers the same value where the values are scalar, so no index is more extreme
    // than the first one that gathers anything: whether that value is the extreme one or only
    // stands in for it, the answer is the same position.
    let Some(values) = arr.flat_values() else {
        return indices
            .into_iter()
            .position(|idx| unsafe { validity.get_unchecked(idx) });
    };

    let mut first_non_null_pos = None;
    for (pos, idx) in indices.into_iter().enumerate() {
        if unsafe { validity.get_unchecked(idx) } {
            if unsafe { values.get_bit_unchecked(idx) } == extreme {
                return Some(pos);
            }
            first_non_null_pos.get_or_insert(pos);
        }
    }
    first_non_null_pos
}

/// [`take_arg_bool_nulls`] for a chunk with no nulls, where position zero stands in for a miss.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
unsafe fn take_arg_bool_no_nulls<I: IntoIterator<Item = usize>>(
    arr: &PlBooleanArray,
    indices: I,
    extreme: bool,
) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    // Every index gathers the same value, so position zero is both the first index that gathers
    // the extremum and the fallback for when none does.
    let Some(values) = arr.flat_values() else {
        return Some(0);
    };

    indices
        .into_iter()
        .position(|idx| unsafe { values.get_bit_unchecked(idx) } == extreme)
        .or(Some(0))
}

/// The position within `indices` of the smallest value they gather.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_arg_min_bool_iter_unchecked_nulls<I: IntoIterator<Item = usize>>(
    arr: &PlBooleanArray,
    indices: I,
) -> Option<usize> {
    unsafe { take_arg_bool_nulls(arr, indices, false) }
}

/// [`take_arg_min_bool_iter_unchecked_nulls`] for a chunk with no nulls in it.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_arg_min_bool_iter_unchecked_no_nulls<I: IntoIterator<Item = usize>>(
    arr: &PlBooleanArray,
    indices: I,
) -> Option<usize> {
    unsafe { take_arg_bool_no_nulls(arr, indices, false) }
}

/// The position within `indices` of the largest value they gather.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_arg_max_bool_iter_unchecked_nulls<I: IntoIterator<Item = usize>>(
    arr: &PlBooleanArray,
    indices: I,
) -> Option<usize> {
    unsafe { take_arg_bool_nulls(arr, indices, true) }
}

/// [`take_arg_max_bool_iter_unchecked_nulls`] for a chunk with no nulls in it.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_arg_max_bool_iter_unchecked_no_nulls<I: IntoIterator<Item = usize>>(
    arr: &PlBooleanArray,
    indices: I,
) -> Option<usize> {
    unsafe { take_arg_bool_no_nulls(arr, indices, true) }
}
