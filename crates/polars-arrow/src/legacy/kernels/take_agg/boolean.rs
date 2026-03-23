#![allow(unsafe_op_in_unsafe_fn)]
use super::*;

/// Take kernel for single chunk and an iterator as index.
/// Returns the position of the minimum value within the group.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_arg_min_bool_iter_unchecked_nulls<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Option<usize> {
    let validity = arr.validity().unwrap();
    let mut first_non_null_pos = None;

    for (pos, idx) in indices.into_iter().enumerate() {
        if validity.get_bit_unchecked(idx) {
            if !arr.value_unchecked(idx) {
                return Some(pos);
            }
            if first_non_null_pos.is_none() {
                first_non_null_pos = Some(pos);
            }
        }
    }
    first_non_null_pos
}

/// Take kernel for single chunk and an iterator as index.
/// Returns the position of the minimum value within the group.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_arg_min_bool_iter_unchecked_no_nulls<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    for (pos, idx) in indices.into_iter().enumerate() {
        if !arr.value_unchecked(idx) {
            return Some(pos);
        }
    }
    Some(0)
}

/// Take kernel for single chunk and an iterator as index.
/// Returns the position of the maximum value within the group.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_arg_max_bool_iter_unchecked_nulls<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Option<usize> {
    let validity = arr.validity().unwrap();
    let mut first_non_null_pos = None;

    for (pos, idx) in indices.into_iter().enumerate() {
        if validity.get_bit_unchecked(idx) {
            if arr.value_unchecked(idx) {
                return Some(pos);
            }
            if first_non_null_pos.is_none() {
                first_non_null_pos = Some(pos);
            }
        }
    }
    first_non_null_pos
}

/// Take kernel for single chunk and an iterator as index.
/// Returns the position of the maximum value within the group.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_arg_max_bool_iter_unchecked_no_nulls<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    for (pos, idx) in indices.into_iter().enumerate() {
        if arr.value_unchecked(idx) {
            return Some(pos);
        }
    }
    Some(0)
}
