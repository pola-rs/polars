#![allow(unsafe_op_in_unsafe_fn)]
//! Gather-and-reduce over a [`PlPrimitiveArray`].

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use either::Either;
use num_traits::ToPrimitive;
use polars_array::PlPrimitiveArray;
use polars_utils::IdxSize;

/// The mask of `arr` as one bit per element, or [`None`] where every element is null.
#[inline]
pub(super) fn flat_validity<T: NativeType>(arr: &PlPrimitiveArray<T>) -> Option<&Bitmap> {
    let validity = arr
        .validity()
        .expect("a chunk with nulls in it holds a validity mask");
    debug_assert!(
        validity.flat_bitmap().is_some() || validity.scalar_value() == Some(false),
        "a mask that repeats a set bit leaves no null for these kernels to be reached with",
    );

    validity.flat_bitmap()
}

/// The values `indices` gather out of a chunk with no nulls in it.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_no_null_primitive_iter_unchecked<
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
>(
    arr: &PlPrimitiveArray<T>,
    indices: I,
) -> impl Iterator<Item = T> {
    debug_assert!(arr.null_count() == 0);

    match arr.scalar_value_ignore_validity() {
        // Every index gathers the one value the buffer holds, so it is read once here rather than
        // through the buffer once per index.
        Some(value) => Either::Left(indices.into_iter().map(move |_| value)),
        None => {
            let values = arr.flat_values().unwrap();
            Either::Right(
                indices
                    .into_iter()
                    .map(|idx| unsafe { *values.get_unchecked(idx) }),
            )
        },
    }
}

/// The non-null values `indices` gather out of a chunk.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked<T: NativeType, I: IntoIterator<Item = usize>>(
    arr: &PlPrimitiveArray<T>,
    indices: I,
) -> impl Iterator<Item = T> {
    let Some(validity) = flat_validity(arr) else {
        // Every element is null, so no index gathers anything.
        return Either::Left(std::iter::empty());
    };

    match arr.scalar_value_ignore_validity() {
        Some(value) => Either::Right(Either::Left(
            indices
                .into_iter()
                .filter(move |&idx| unsafe { validity.get_bit_unchecked(idx) })
                .map(move |_| value),
        )),
        None => {
            let values = arr.flat_values().unwrap();
            Either::Right(Either::Right(
                indices
                    .into_iter()
                    .filter(|&idx| unsafe { validity.get_bit_unchecked(idx) })
                    .map(|idx| unsafe { *values.get_unchecked(idx) }),
            ))
        },
    }
}

/// Folds the values `indices` read through `value_at` with `f`, counting the nulls it skips.
#[inline]
fn fold_gathered<T, TOut>(
    indices: impl IntoIterator<Item = usize>,
    validity: &Bitmap,
    value_at: impl Fn(usize) -> T,
    init: TOut,
    f: impl Fn(TOut, T) -> TOut,
) -> (TOut, IdxSize) {
    let mut null_count = 0 as IdxSize;
    let out = indices.into_iter().fold(init, |acc, idx| {
        if unsafe { validity.get_bit_unchecked(idx) } {
            f(acc, value_at(idx))
        } else {
            null_count += 1;
            acc
        }
    });

    (out, null_count)
}

/// Folds the non-null values `indices` gather with `f`, alongside the number of nulls skipped.
///
/// # Safety
/// Every index must be in bounds of `arr`.
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked_count_nulls<
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
    TOut,
    F: Fn(TOut, T) -> TOut,
>(
    arr: &PlPrimitiveArray<T>,
    indices: I,
    init: TOut,
    f: F,
    len: IdxSize,
) -> Option<(TOut, IdxSize)> {
    let Some(validity) = flat_validity(arr) else {
        // Every element is null, so every one of the `len` indices gathered one.
        return None;
    };

    // Which buffer the values come out of is settled once, ahead of the fold: every index of a
    // scalar chunk gathers the one value it holds.
    let (out, null_count) = match arr.scalar_value_ignore_validity() {
        Some(value) => fold_gathered(indices, validity, |_| value, init, f),
        None => {
            let values = arr.flat_values().unwrap();
            let value_at = |idx| unsafe { *values.get_unchecked(idx) };
            fold_gathered(indices, validity, value_at, init, f)
        },
    };

    if null_count == len {
        None
    } else {
        Some((out, null_count))
    }
}
