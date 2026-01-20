#![allow(unsafe_op_in_unsafe_fn)]
//! kernels that combine take and aggregations.
mod boolean;
mod var;

pub use boolean::*;
use num_traits::ToPrimitive;
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
    I: IntoIterator<Item = usize>,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> impl Iterator<Item = T> {
    debug_assert!(arr.null_count() == 0);
    let array_values = arr.values().as_slice();

    indices
        .into_iter()
        .map(|idx| *array_values.get_unchecked(idx))
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked<T: NativeType, I: IntoIterator<Item = usize>>(
    arr: &PrimitiveArray<T>,
    indices: I,
) -> impl Iterator<Item = T> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().unwrap();

    indices
        .into_iter()
        .filter(|&idx| validity.get_bit_unchecked(idx))
        .map(|idx| *array_values.get_unchecked(idx))
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked_count_nulls<
    T: NativeType + ToPrimitive,
    I: IntoIterator<Item = usize>,
    TOut,
    F: Fn(TOut, T) -> TOut,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
    init: TOut,
    f: F,
    len: IdxSize,
) -> Option<(TOut, IdxSize)> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().expect("null buffer should be there");

    let mut null_count = 0 as IdxSize;
    let out = indices.into_iter().fold(init, |acc, idx| {
        if validity.get_bit_unchecked(idx) {
            f(acc, *array_values.get_unchecked(idx))
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
) -> Option<&'a [u8]> {
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

/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_bin_iter_unchecked_arg<
    'a,
    I: IntoIterator<Item = usize>,
    F: Fn((IdxSize, &'a [u8]), (IdxSize, &'a [u8])) -> (IdxSize, &'a [u8]),
>(
    arr: &'a BinaryViewArray,
    indices: I,
    f: F,
) -> Option<IdxSize> {
    let validity = arr.validity().unwrap();

    indices
        .into_iter()
        .enumerate()
        .filter_map(|(pos, idx)| {
            if validity.get_bit_unchecked(idx) {
                Some((pos as IdxSize, arr.value_unchecked(idx)))
            } else {
                None
            }
        })
        .reduce(f)
        .map(|(pos, _)| pos)
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
) -> Option<&'a [u8]> {
    indices
        .into_iter()
        .map(|idx| arr.value_unchecked(idx))
        .reduce(|acc, str_val| f(acc, str_val))
}

/// # Safety
/// caller must ensure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_bin_iter_unchecked_no_null_arg<
    'a,
    I: IntoIterator<Item = usize>,
    F: Fn((IdxSize, &'a [u8]), (IdxSize, &'a [u8])) -> (IdxSize, &'a [u8]),
>(
    arr: &'a BinaryViewArray,
    indices: I,
    f: F,
) -> Option<IdxSize> {
    indices
        .into_iter()
        .enumerate()
        .map(|(pos, idx)| (pos as IdxSize, arr.value_unchecked(idx)))
        .reduce(f)
        .map(|(pos, _)| pos)
}
