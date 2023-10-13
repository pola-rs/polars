use super::*;

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_min_bool_iter_unchecked_nulls<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
    len: IdxSize,
) -> Option<bool> {
    let mut null_count = 0 as IdxSize;
    let validity = arr.validity().unwrap();

    for idx in indices {
        if validity.get_bit_unchecked(idx) {
            if !arr.value_unchecked(idx) {
                return Some(false);
            }
        } else {
            null_count += 1;
        }
    }
    if null_count == len {
        None
    } else {
        Some(true)
    }
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_min_bool_iter_unchecked_no_nulls<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Option<bool> {
    if arr.is_empty() {
        return None;
    }

    for idx in indices {
        if !arr.value_unchecked(idx) {
            return Some(false);
        }
    }
    Some(true)
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_max_bool_iter_unchecked_nulls<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
    len: IdxSize,
) -> Option<bool> {
    let mut null_count = 0 as IdxSize;
    let validity = arr.validity().unwrap();

    for idx in indices {
        if validity.get_bit_unchecked(idx) {
            if arr.value_unchecked(idx) {
                return Some(true);
            }
        } else {
            null_count += 1;
        }
    }
    if null_count == len {
        None
    } else {
        Some(false)
    }
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_max_bool_iter_unchecked_no_nulls<I: IntoIterator<Item = usize>>(
    arr: &BooleanArray,
    indices: I,
) -> Option<bool> {
    if arr.is_empty() {
        return None;
    }

    for idx in indices {
        if arr.value_unchecked(idx) {
            return Some(true);
        }
    }
    Some(false)
}
