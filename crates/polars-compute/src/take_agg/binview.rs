#![allow(unsafe_op_in_unsafe_fn)]
//! Gather-and-reduce over a [`PlBinaryViewArray`].

use polars_array::PlBinaryViewArray;
use polars_utils::IdxSize;

/// The bytes every index gathers, where the views buffer holds a single view.
#[inline]
fn repeated_value(arr: &PlBinaryViewArray) -> Option<&[u8]> {
    // SAFETY: a scalar views buffer is only held by a non-empty array, so element zero is in
    // bounds.
    arr.views_are_scalar()
        .then(|| unsafe { arr.value_unchecked(0) })
}

/// Folds the non-null values `indices` gather with `f`.
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
