#![allow(unsafe_op_in_unsafe_fn)]
//! Gather-and-reduce over a [`PlBooleanArray`].
//!
//! These answer the *position within the group* of an extremum, so they walk the indices in order
//! and stop at the first one that gathers the extreme value. A values bitmap of a single bit makes
//! every index gather that one value, so the walk has nothing to compare and the first index that
//! gathers anything at all is the answer.

use polars_array::PlBooleanArray;

/// The values bitmap of `arr`, which holds one bit per element because it is not scalar.
#[inline]
fn flat_values(arr: &PlBooleanArray) -> &arrow::bitmap::Bitmap {
    arr.flat_values()
        .expect("a values bitmap that is not scalar holds one bit per element")
}

/// The position in `indices` of the first index that gathers `extreme`, or of the first that
/// gathers a non-null value at all.
///
/// This is the shape of both the arg-min and the arg-max of a boolean: the smaller of the two
/// values is `false` and the larger is `true`, so `extreme` is all that differs.
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

    match arr.scalar_values() {
        // Every index gathers the same value, so no index is more extreme than the first one that
        // gathers anything: whether that value is the extreme one or only stands in for it, the
        // answer is the same position.
        Some(_) => indices
            .into_iter()
            .position(|idx| unsafe { validity.get_unchecked(idx) }),
        None => {
            let values = flat_values(arr);
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
        },
    }
}

/// [`take_arg_bool_nulls`] for a chunk with no nulls in it, where every index gathers a value and
/// so position zero stands in wherever no index gathers the extreme one.
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
    if arr.scalar_values().is_some() {
        return Some(0);
    }

    let values = flat_values(arr);
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

#[cfg(test)]
mod tests {
    use super::*;

    const LENGTH: usize = 6;
    const INDICES: [usize; 4] = [0, 3, 1, 5];

    fn scalar_and_flat(value: bool, mask: Option<[bool; LENGTH]>) -> [PlBooleanArray; 2] {
        let validity = mask.map(|mask| mask.into_iter().collect());
        [
            PlBooleanArray::new_scalar(value, LENGTH).with_validity(validity.clone()),
            PlBooleanArray::from_values(std::iter::repeat_n(value, LENGTH).collect())
                .with_validity(validity),
        ]
    }

    #[test]
    fn a_repeated_value_gathers_the_same_either_way() {
        for value in [false, true] {
            let [scalar, flat] = scalar_and_flat(value, None);

            assert_eq!(
                unsafe { take_arg_min_bool_iter_unchecked_no_nulls(&scalar, INDICES) },
                unsafe { take_arg_min_bool_iter_unchecked_no_nulls(&flat, INDICES) },
            );
            assert_eq!(
                unsafe { take_arg_max_bool_iter_unchecked_no_nulls(&scalar, INDICES) },
                unsafe { take_arg_max_bool_iter_unchecked_no_nulls(&flat, INDICES) },
            );
        }
    }

    #[test]
    fn a_repeated_null_gathers_nothing() {
        let arr = PlBooleanArray::new_full_null(LENGTH);
        assert_eq!(
            unsafe { take_arg_min_bool_iter_unchecked_nulls(&arr, INDICES) },
            None,
        );
        assert_eq!(
            unsafe { take_arg_max_bool_iter_unchecked_nulls(&arr, INDICES) },
            None,
        );
    }

    #[test]
    fn a_repeated_value_under_a_flat_mask() {
        // The first of `INDICES` is masked out, so the answer is a later position.
        let mask = [false, true, true, true, true, true];
        for value in [false, true] {
            let [scalar, flat] = scalar_and_flat(value, Some(mask));

            let min = unsafe { take_arg_min_bool_iter_unchecked_nulls(&scalar, INDICES) };
            assert_eq!(min, Some(1));
            assert_eq!(min, unsafe {
                take_arg_min_bool_iter_unchecked_nulls(&flat, INDICES)
            });

            let max = unsafe { take_arg_max_bool_iter_unchecked_nulls(&scalar, INDICES) };
            assert_eq!(max, Some(1));
            assert_eq!(max, unsafe {
                take_arg_max_bool_iter_unchecked_nulls(&flat, INDICES)
            });
        }
    }

    /// A flat chunk that does hold both values still picks the extreme one out.
    #[test]
    fn a_flat_chunk_finds_the_extremum() {
        let arr = PlBooleanArray::from_values(
            [true, true, false, true, false, true].into_iter().collect(),
        );
        // `INDICES` is [0, 3, 1, 5]: every one of those is `true`, so the min falls back to zero.
        assert_eq!(
            unsafe { take_arg_min_bool_iter_unchecked_no_nulls(&arr, INDICES) },
            Some(0),
        );
        assert_eq!(
            unsafe { take_arg_max_bool_iter_unchecked_no_nulls(&arr, INDICES) },
            Some(0),
        );
        // Index 2 gathers `false`, at position 1 of these indices.
        assert_eq!(
            unsafe { take_arg_min_bool_iter_unchecked_no_nulls(&arr, [0usize, 2, 3]) },
            Some(1),
        );
    }
}
