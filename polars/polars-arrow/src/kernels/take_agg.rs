//! kernels that combine take and aggregations.
use crate::array::PolarsArray;
use crate::index::IdxSize;
use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use num::{NumCast, ToPrimitive};

/// Take kernel for single chunk without nulls and an iterator as index.
/// # Safety
/// caller must enure iterators indexes are in bounds
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
    init: TOut,
) -> TOut {
    debug_assert!(!arr.has_validity());
    let array_values = arr.values().as_slice();

    indices.into_iter().fold(init, |acc, idx| {
        f(
            acc,
            NumCast::from(*array_values.get_unchecked(idx)).unwrap_unchecked(),
        )
    })
}

/// Take kernel for single chunk and an iterator as index.
/// # Safety
/// caller must enure iterators indexes are in bounds
#[inline]
pub unsafe fn take_agg_primitive_iter_unchecked<
    T: NativeType,
    I: IntoIterator<Item = usize>,
    F: Fn(T, T) -> T,
>(
    arr: &PrimitiveArray<T>,
    indices: I,
    f: F,
    init: T,
) -> Option<T> {
    let array_values = arr.values().as_slice();
    let validity = arr.validity().expect("null buffer should be there");

    let out = indices.into_iter().fold(init, |acc, idx| {
        if validity.get_bit_unchecked(idx) {
            f(acc, *array_values.get_unchecked(idx))
        } else {
            acc
        }
    });
    if out == init {
        None
    } else {
        Some(out)
    }
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
    if out == init {
        None
    } else {
        Some((out, null_count))
    }
}
