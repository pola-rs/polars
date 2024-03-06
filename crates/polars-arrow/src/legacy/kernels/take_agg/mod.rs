//! kernels that combine take and aggregations.
mod boolean;
mod var;

pub use boolean::*;
use num_traits::{NumCast, ToPrimitive};
use polars_utils::IdxSize;
pub use var::*;

use crate::array::{Array, BinaryViewArray, BooleanArray, PrimitiveArray};
use crate::types::NativeType;

/// Take kernel for single chunk without nulls and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_no_null_primitive_iter_unchecked<
    T: NativeType + ToPrimitive,
    TOut: NumCast + NativeType,
    I: IntoIterator<Item = usize>,
    F: Fn(TOut, TOut) -> TOut,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
    f: F,
) -> Option<TOut> {
    debug_assert!(arr.null_count() == 0);
    let array_values = arr.values().as_slice();

    indices
        .into_iter()
        .map(|idx| TOut::from(*array_values.get_unchecked(idx)).unwrap_unchecked())
        .reduce(f)
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked<
    T: NativeType,
    I: IntoIterator<Item = usize>,
    F: Fn(T, T) -> T,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
    f: F,
) -> Option<T> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().unwrap();

    indices
        .into_iter()
        .filter(|&idx| validity.get_bit_unchecked(idx))
        .map(|idx| *array_values.get_unchecked(idx))
        .reduce(f)
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must enure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked_count_nulls<
    T: NativeType + ToPrimitive,
    TOut: NumCast + NativeType,
    I: IntoIterator<Item = usize>,
    F: Fn(TOut, TOut) -> TOut,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
    f: F,
    init: TOut,
    len: IdxSize,
) -> Option<(TOut, IdxSize)> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().expect("null buffer should be there");

    let mut null_count = 0 as IdxSize;
    let out = indices.into_iter().fold(init, |acc, idx| {
        if validity.get_bit_unchecked(idx) {
            f(
                acc,
                NumCast::from(*array_values.get_unchecked(idx)).unwrap_unchecked(),
            )
        } else {
            null_count += 1;
            acc
        }
    });
    if null_count == len {
        None
    } else {
        Some((out, null_count))
    }
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_bin_iter_unchecked<
    'a,
    I: IntoIterator<Item = usize>,
    F: Fn(&'a [u8], &'a [u8]) -> &'a [u8],
>(
    arr: &'a BinaryViewArray,
    indices: I,
    f: F,
    len: IdxSize,
) -> Option<&[u8]> {
    let mut null_count = 0 as IdxSize;
    let validity = arr.validity().unwrap();

    let out = indices
        .into_iter()
        .map(|idx| {
            if validity.get_bit_unchecked(idx) {
                Some(arr.value_unchecked(idx))
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

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_bin_iter_unchecked_no_null<
    'a,
    I: IntoIterator<Item = usize>,
    F: Fn(&'a [u8], &'a [u8]) -> &'a [u8],
>(
    arr: &'a BinaryViewArray,
    indices: I,
    f: F,
) -> Option<&[u8]> {
    indices
        .into_iter()
        .map(|idx| arr.value_unchecked(idx))
        .reduce(|acc, str_val| f(acc, str_val))
}
