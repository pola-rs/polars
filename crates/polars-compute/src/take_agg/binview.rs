#![allow(unsafe_op_in_unsafe_fn)]
//! Gather-and-reduce over a [`PlBinaryViewArray`].
//!
//! A views buffer of a single view makes every index gather the same bytes, so a fold over them
//! reads those bytes once and never touches the buffer again.

use polars_array::PlBinaryViewArray;
use polars_utils::IdxSize;

/// The bytes every index gathers, where the views buffer holds a single view.
#[inline]
fn repeated_value(arr: &PlBinaryViewArray) -> Option<&[u8]> {
    // SAFETY: the array is not empty, so element zero is in bounds.
    (arr.views_are_scalar() && !arr.is_empty()).then(|| unsafe { arr.value_unchecked(0) })
}

/// Folds the non-null values `indices` gather with `f`.
///
/// Returns [`None`] where every one of the `len` indices gathered a null.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_bin_iter_unchecked<
    'a,
    I: IntoIterator<Item = usize>,
    F: Fn(&'a [u8], &'a [u8]) -> &'a [u8],
>(
    arr: &'a PlBinaryViewArray,
    indices: I,
    f: F,
    len: IdxSize,
) -> Option<&'a [u8]> {
    let validity = arr
        .validity()
        .expect("a chunk with nulls in it holds a validity mask");

    // Every element is null, so every one of the `len` indices gathered one.
    if validity.scalar_value() == Some(false) {
        return None;
    }

    let repeated = repeated_value(arr);
    let mut null_count = 0 as IdxSize;

    let out = indices
        .into_iter()
        .map(|idx| {
            if unsafe { validity.get_unchecked(idx) } {
                Some(match repeated {
                    Some(bytes) => bytes,
                    None => unsafe { arr.value_unchecked(idx) },
                })
            } else {
                None
            }
        })
        .reduce(|acc, opt_val| match (acc, opt_val) {
            (Some(acc), Some(str_val)) => Some(f(acc, str_val)),
            (_, None) => {
                null_count += 1;
                acc
            },
            (None, Some(str_val)) => Some(str_val),
        });

    if null_count == len {
        None
    } else {
        out.flatten()
    }
}

/// The position within `indices` that `f` folds down to, over the non-null values they gather.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_bin_iter_unchecked_arg<
    'a,
    I: IntoIterator<Item = usize>,
    F: Fn((IdxSize, &'a [u8]), (IdxSize, &'a [u8])) -> (IdxSize, &'a [u8]),
>(
    arr: &'a PlBinaryViewArray,
    indices: I,
    f: F,
) -> Option<IdxSize> {
    let validity = arr
        .validity()
        .expect("a chunk with nulls in it holds a validity mask");

    // Every element is null, so no index gathers anything.
    if validity.scalar_value() == Some(false) {
        return None;
    }

    let repeated = repeated_value(arr);

    indices
        .into_iter()
        .enumerate()
        .filter_map(|(pos, idx)| {
            if unsafe { validity.get_unchecked(idx) } {
                let bytes = match repeated {
                    Some(bytes) => bytes,
                    None => unsafe { arr.value_unchecked(idx) },
                };
                Some((pos as IdxSize, bytes))
            } else {
                None
            }
        })
        .reduce(f)
        .map(|(pos, _)| pos)
}

/// [`take_agg_bin_iter_unchecked`] for a chunk with no nulls in it.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_bin_iter_unchecked_no_null<
    'a,
    I: IntoIterator<Item = usize>,
    F: Fn(&'a [u8], &'a [u8]) -> &'a [u8],
>(
    arr: &'a PlBinaryViewArray,
    indices: I,
    f: F,
) -> Option<&'a [u8]> {
    // Every index gathers the same bytes, which are read once here: the fold runs over them
    // without the buffer being touched again.
    if let Some(bytes) = repeated_value(arr) {
        return indices.into_iter().map(|_| bytes).reduce(&f);
    }

    indices
        .into_iter()
        .map(|idx| unsafe { arr.value_unchecked(idx) })
        .reduce(|acc, str_val| f(acc, str_val))
}

/// [`take_agg_bin_iter_unchecked_arg`] for a chunk with no nulls in it.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_bin_iter_unchecked_no_null_arg<
    'a,
    I: IntoIterator<Item = usize>,
    F: Fn((IdxSize, &'a [u8]), (IdxSize, &'a [u8])) -> (IdxSize, &'a [u8]),
>(
    arr: &'a PlBinaryViewArray,
    indices: I,
    f: F,
) -> Option<IdxSize> {
    let repeated = repeated_value(arr);

    indices
        .into_iter()
        .enumerate()
        .map(|(pos, idx)| {
            let bytes = match repeated {
                Some(bytes) => bytes,
                None => unsafe { arr.value_unchecked(idx) },
            };
            (pos as IdxSize, bytes)
        })
        .reduce(f)
        .map(|(pos, _)| pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LENGTH: usize = 6;
    const INDICES: [usize; 4] = [0, 3, 1, 5];

    fn min<'a>(a: &'a [u8], b: &'a [u8]) -> &'a [u8] {
        if b < a { b } else { a }
    }

    fn min_arg<'a>(a: (IdxSize, &'a [u8]), b: (IdxSize, &'a [u8])) -> (IdxSize, &'a [u8]) {
        if b.1 < a.1 { b } else { a }
    }

    fn scalar_and_flat(value: &[u8], mask: Option<[bool; LENGTH]>) -> [PlBinaryViewArray; 2] {
        let validity: Option<arrow::bitmap::Bitmap> = mask.map(|mask| mask.into_iter().collect());
        [
            PlBinaryViewArray::new_scalar(value, LENGTH).with_validity(validity.clone()),
            PlBinaryViewArray::from_values_iter(std::iter::repeat_n(value, LENGTH))
                .with_validity(validity),
        ]
    }

    #[test]
    fn a_repeated_value_gathers_the_same_either_way() {
        let [scalar, flat] = scalar_and_flat(b"repeated", None);

        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_no_null(&scalar, INDICES, min) },
            Some(&b"repeated"[..]),
        );
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_no_null(&scalar, INDICES, min) },
            unsafe { take_agg_bin_iter_unchecked_no_null(&flat, INDICES, min) },
        );
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_no_null_arg(&scalar, INDICES, min_arg) },
            unsafe { take_agg_bin_iter_unchecked_no_null_arg(&flat, INDICES, min_arg) },
        );

        // No index at all gathers nothing, whatever the representation.
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_no_null(&scalar, [], min) },
            None,
        );
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_no_null_arg(&scalar, [], min_arg) },
            None,
        );
    }

    #[test]
    fn a_repeated_null_gathers_nothing() {
        let arr = PlBinaryViewArray::new_full_null(LENGTH);
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked(&arr, INDICES, min, INDICES.len() as IdxSize) },
            None,
        );
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_arg(&arr, INDICES, min_arg) },
            None,
        );
    }

    #[test]
    fn a_repeated_value_under_a_flat_mask() {
        let mask = [false, true, true, true, true, true];
        let [scalar, flat] = scalar_and_flat(b"repeated", Some(mask));

        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked(&scalar, INDICES, min, INDICES.len() as IdxSize) },
            unsafe { take_agg_bin_iter_unchecked(&flat, INDICES, min, INDICES.len() as IdxSize) },
        );
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_arg(&scalar, INDICES, min_arg) },
            unsafe { take_agg_bin_iter_unchecked_arg(&flat, INDICES, min_arg) },
        );
        // The first index is masked out, so the argument is a later position.
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_arg(&scalar, INDICES, min_arg) },
            Some(1),
        );
    }

    /// A flat chunk that holds distinct values still folds down to the extreme one.
    #[test]
    fn a_flat_chunk_finds_the_extremum() {
        let arr = PlBinaryViewArray::from_values_iter(
            [b"d".as_slice(), b"b", b"e", b"a", b"c", b"f"].into_iter(),
        );
        // `INDICES` is [0, 3, 1, 5]: "d", "a", "b", "f".
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_no_null(&arr, INDICES, min) },
            Some(&b"a"[..]),
        );
        assert_eq!(
            unsafe { take_agg_bin_iter_unchecked_no_null_arg(&arr, INDICES, min_arg) },
            Some(1),
        );
    }
}
